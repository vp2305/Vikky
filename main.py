from typing import Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
from interface import tts, vad, asr
from jinja2 import Template
from Levenshtein import distance
from sounddevice import CallbackFlags

import copy
import datetime
import time
import yaml
import sys
import queue
import threading
import re
import numpy as np
import requests
import json
import sounddevice as sd

logger.remove(0)
logger.add(
    sys.stderr,
    level="INFO",
)

ASR_MODEL = "ggml-medium-32-2.en.bin"
VAD_MODEL = "silero_vad.onnx"
VIKKY_CONFIG_PATH = "vikky_config.yml"

LLM_STOP_SEQUENCE = "<|eot_id|>"  # End of sentence token for Meta-Llama-3
LLAMA3_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

PAUSE_TIME = 0.05  # Time to wait between processing loops
VAD_SIZE = 90  # Milliseconds of sample for Voice Activity Detection (VAD)
VAD_THRESHOLD = 0.9  # Threshold for VAD detection
PAUSE_LIMIT = 1000  # Milliseconds of pause allowed before processing
SIMILARITY_THRESHOLD = 2  # Threshold for wake word similarity
BUFFER_SIZE = 600  # Milliseconds of buffer before VAD detection

DEFAULT_PERSONALITY_PREPROMPT = {
    "role": "system",
    "content": "You are a helpful AI assistant. You are here to assist the user in their tasks.",
}


@dataclass
class VikkyConfig:
    completion_url: str
    wake_word: Optional[str]
    announcement: Optional[str]
    personality_preprompt: List[dict[str, str]]
    interruptible: bool
    voice_model: str = "glados.onnx"

    @classmethod
    def from_yaml(cls, path: str, key_to_config: Sequence[str] | None = ("Vikky",)):
        key_to_config = key_to_config or []

        with open(path, "r") as file:
            data = yaml.safe_load(file)

        config = data
        for nested_key in key_to_config:
            config = config[nested_key]

        return cls(**config)


class Vikky:
    def __init__(
        self,
        voice_model: str,
        completion_url: str,
        wake_word: str | None = None,
        announcement: str | None = None,
        personality_preprompt: Sequence[dict[str, str]] = DEFAULT_PERSONALITY_PREPROMPT,
        interruptible: bool = False,
    ) -> None:
        """
        Initializes the VoiceRecognition class, setting up necessary models, streams, and queues.

        This class is not thread-safe, so you should only use it from one thread. It works like this:
        1. The audio stream is continuously listening for input.
        2. The audio is buffered until voice activity is detected. This is to make sure that the entire sentence is captured, including before voice activity is detected.
        2. While voice activity is detected, the audio is stored, together with the buffered audio.
        3. When voice activity is not detected after a short time (the PAUSE_LIMIT), the audio is transcribed. If voice is detected again during this time, the timer is reset and the recording continues.
        4. After the voice stops, the listening stops, and the audio is transcribed.
        5. If a wake word is set, the transcribed text is checked for similarity to the wake word.
        6. The function is called with the transcribed text as the argument.
        7. The audio stream is reset (buffers cleared), and listening continues.

        Args:
            wake_word (str, optional): The wake word to use for activation. Defaults to None.
        """
        self.completion_url = completion_url
        self.wake_word = wake_word

        self._vad_model = vad.VAD(model_path=str(Path.cwd() / "models" / VAD_MODEL))
        self._asr_model = asr.ASR(model=str(Path.cwd() / "models" / ASR_MODEL))
        self._tts = tts.Synthesizer(
            model_path=str(Path.cwd() / "models" / voice_model), use_cuda=False
        )

        self._samples: List[np.ndarray] = []
        self._sample_queue: queue.Queue[Tuple[np.ndarray, np.ndarray]] = queue.Queue()
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=BUFFER_SIZE // VAD_SIZE
        )
        self._recording_started = False
        self._gap_counter = 0

        self._messages = personality_preprompt
        self.llm_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()

        self.processing = False
        self.currently_speaking = False
        self.interruptible = interruptible

        self.shutdown_event = threading.Event()  # Event to signal shutdown
        self.template = Template(LLAMA3_TEMPLATE)

        # Thread to process messages
        llm_thread = threading.Thread(target=self.process_LLM)
        llm_thread.start()

        # Thread to process TTS
        tts_thread = threading.Thread(target=self.process_TTS_thread)
        tts_thread.start()

        if announcement:
            self.handle_announcement(announcement)

        # signature defined by sd.InputStream, see docstring of callback there
        # noinspection PyUnusedLocal
        def audio_callback_for_sdInputStream(
            indata: np.ndarray, frames: int, time: Any, status: CallbackFlags
        ):
            data = indata.copy().squeeze()  # Reduce to single channel if necessary
            vad_confidence = self._vad_model.process_chunk(data) > VAD_THRESHOLD
            self._sample_queue.put((data, vad_confidence))

        self.input_stream = sd.InputStream(
            samplerate=self._tts.rate,
            channels=1,
            callback=audio_callback_for_sdInputStream,
            blocksize=int(self._tts.rate * VAD_SIZE / 1000),
        )

    @property
    def messages(self) -> Sequence[dict[str, str]]:
        return self._messages

    @classmethod
    def from_config(cls, config: VikkyConfig):

        personality_preprompt = []
        for line in config.personality_preprompt:
            personality_preprompt.append(
                {"role": list(line.keys())[0], "content": list(line.values())[0]}
            )

        return cls(
            voice_model=config.voice_model,
            completion_url=config.completion_url,
            wake_word=config.wake_word,
            personality_preprompt=personality_preprompt,
            announcement=config.announcement,
            interruptible=config.interruptible,
        )

    @classmethod
    def from_yaml(cls, path: str):
        return cls.from_config(VikkyConfig.from_yaml(path))

    def handle_announcement(self, announcement: str):
        """
        Handles the announcement audio.

        Args:
            announcement (str): The announcement to be played.
        """
        currentTime = datetime.datetime.now()
        announcement = announcement.replace(
            "<GREETINGS>",
            (
                "Good Morning"
                if 5 <= currentTime.hour < 12
                else (
                    "Good Afternoon" if 12 <= currentTime.hour < 18 else "Good Evening"
                )
            ),
        )
        announcement = (
            announcement.replace("<TIME>", currentTime.strftime("%I %p"))
            .replace("AM", "A.M.")
            .replace("PM", "P.M.")
        )
        audio = self._tts.generate_speech_audio(announcement)
        logger.info(f"TTS text: {announcement}")
        sd.play(
            audio,
            self._tts.rate,
        )
        sd.wait()

    def start_listen_event_loop(self):
        """
        Starts the Vikky voice assistant, continuously listening for input and responding.
        """
        self.input_stream.start()
        logger.success("Audio Module Operational")
        logger.success("Listening...")

        # Loop forever, but is 'paused' when new samples are not available
        try:
            while True:
                sample, vad_confidence = self._sample_queue.get()
                self._handle_audio_sample(sample, vad_confidence)
        except KeyboardInterrupt:
            self.shutdown_event.set()
            self.input_stream.stop()

    def _handle_audio_sample(self, sample: np.ndarray, vad_confidence: bool):
        """
        Handles the processing of each audio sample.

        If the recording has not started, the sample is added to the circular buffer.

        If the recording has started, the sample is added to the samples list, and the pause
        limit is checked to determine when to process the detected audio.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        if not self._recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample: np.ndarray, vad_confidence: bool):
        """
        Manages the circular buffer of audio samples before activation (i.e., before the voice is detected).

        If the buffer is full, the oldest sample is discarded to make room for new ones.

        If voice activity is detected, the audio stream is stopped, and the processing is turned off to prevent overlap with the LLM and TTS threads.

        Args:
            sample (np.ndarray): The audio sample to process.
            vad_confidence (bool): Whether voice activity is detected in the sample.
        """
        if self._buffer.full():
            self._buffer.get()  # Discard the oldest sample to make room for new ones
        self._buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            sd.stop()  # Stop the audio stream to prevent overlap
            self.processing = (
                False  # Turns off processing on threads for the LLM and TTS!!!
            )
            self._samples = list(self._buffer.queue)
            self._recording_started = True

    def _process_activated_audio(self, sample: np.ndarray, vad_confidence: bool):
        """
        Processes audio samples after activation (i.e., after the wake word is detected).

        Uses a pause limit to determine when to process the detected audio. This is to
        ensure that the entire sentence is captured before processing, including slight gaps.
        """

        self._samples.append(sample)

        if not vad_confidence:
            self._gap_counter += 1
            if self._gap_counter >= PAUSE_LIMIT // VAD_SIZE:
                self._process_detected_audio()
        else:
            self._gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Calculates the nearest Levenshtein distance from the detected text to the wake word.

        This is used as 'wake_word' is not a common word, and Whisper can sometimes mishear it.
        """
        assert self.wake_word is not None, "Wake word should not be None"

        words = text.split()
        closest_distance = min(
            [distance(word.lower(), self.wake_word) for word in words]
        )
        return closest_distance < SIMILARITY_THRESHOLD

    def _process_detected_audio(self):
        """
        Processes the detected audio and generates a response.

        This function is called when the pause limit is reached after the voice stops.
        It transcribes the audio and checks for the wake word if it is set. If the wake
        word is detected, the detected text is sent to the LLM model for processing.
        The audio stream is then reset, and listening continues.
        """
        logger.info("Detected pause after speech. Processing...")
        self.input_stream.stop()

        detected_text = self.asr(self._samples)

        if detected_text:
            logger.success(f"ASR text: '{detected_text}'")

            if self.wake_word and not self._wakeword_detected(detected_text):
                logger.info(f"Required wake word {self.wake_word=} not detected.")
            else:
                self.llm_queue.put(detected_text)
                self.processing = True
                self.currently_speaking = True

        if not self.interruptible:
            while self.currently_speaking:
                time.sleep(PAUSE_TIME)

        self.reset()
        self.input_stream.start()

    def asr(self, samples: List[np.ndarray]) -> str:
        """
        Performs automatic speech recognition on the collected samples.
        """
        audio = np.concatenate(samples)
        logger.info(f"Audio: {audio}")

        detected_text = self._asr_model.transcribe(audio)
        return detected_text

    def reset(self):
        """
        Resets the recording state and clears buffers.
        """
        logger.info("Resetting recorder...")
        self._recording_started = False
        self._samples.clear()
        self._gap_counter = 0
        with self._buffer.mutex:
            self._buffer.queue.clear()

    def process_TTS_thread(self):
        """
        Processes the LLM generated text using the TTS model.

        Runs in a separate thread to allow for continuous processing of the LLM output.
        """
        assistant_text = (
            []
        )  # The text generated by the assistant, to be spoken by the TTS
        system_text = (
            []
        )  # The text logged to the system prompt when the TTS is interrupted
        finished = False  # a flag to indicate when the TTS has finished speaking
        interrupted = (
            False  # a flag to indicate when the TTS was interrupted by new input
        )

        while not self.shutdown_event.is_set():
            try:
                generated_text = self.tts_queue.get(timeout=PAUSE_TIME)

                if (
                    generated_text == "<EOS>"
                ):  # End of stream token generated in process_LLM_thread
                    finished = True
                elif not generated_text:
                    # should not happen!
                    logger.warning("Empty string sent to TTS")
                else:
                    logger.success(f"TTS text: {generated_text}")
                    audio = self._tts.generate_speech_audio(generated_text)
                    total_samples = len(audio)

                    if total_samples:
                        sd.play(audio, self._tts.rate)

                        interrupted, percentage_played = self.percentage_played(
                            total_samples
                        )

                        if interrupted:
                            clipped_text = self.clip_interrupted_sentence(
                                generated_text, percentage_played
                            )

                            logger.info(
                                f"TTS interrupted at {percentage_played}%: {clipped_text}"
                            )
                            system_text = copy.deepcopy(assistant_text)
                            system_text.append(clipped_text)
                            finished = True

                        assistant_text.append(generated_text)

                if finished:
                    self.messages.append(
                        {"role": "assistant", "content": " ".join(assistant_text)}
                    )
                    # if interrupted:
                    #     self.messages.append(
                    #         {
                    #             "role": "system",
                    #             "content": f"USER INTERRUPTED {self.wake_word}, TEXT DELIVERED: {' '.join(system_text)}",
                    #         }
                    #     )
                    assistant_text = []
                    finished = False
                    interrupted = False
                    self.currently_speaking = False

            except queue.Empty:
                pass

    def clip_interrupted_sentence(
        self, generated_text: str, percentage_played: float
    ) -> str:
        """
        Clips the generated text if the TTS was interrupted.

        Args:

            generated_text (str): The generated text from the LLM model.
            percentage_played (float): The percentage of the audio played before the TTS was interrupted.

            Returns:

            str: The clipped text.

        """
        tokens = generated_text.split()
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])

        # If the TTS was cut off, make that clear
        if words_to_print < len(tokens):
            text = text + "<INTERRUPTED>"
        return text

    def percentage_played(self, total_samples: int) -> Tuple[bool, int]:
        interrupted = False
        start_time = time.time()
        played_samples = 0.0

        while sd.get_stream().active:
            # Should the TTS stream should still be active?
            time.sleep(PAUSE_TIME)
            if self.processing is False:
                sd.stop()  # Stop the audio stream
                self.tts_queue = queue.Queue()  # Clear the TTS queue
                interrupted = True
                break

        elapsed_time = (
            time.time() - start_time + 0.12
        )  # slight delay to ensure all audio timing is correct
        played_samples = elapsed_time * self._tts.rate

        # Calculate percentage of audio played
        percentage_played = min(int((played_samples / total_samples * 100)), 100)
        return interrupted, percentage_played

    def process_LLM(self):
        """
        Processes the detected text using the LLM model.

        """
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_queue.get(timeout=0.1)

                self.messages.append({"role": "user", "content": detected_text})

                prompt = self.template.render(
                    messages=self.messages,
                    bos_token="<|begin_of_text|>",
                    add_generation_prompt=True,
                )

                logger.debug(f"{prompt=}")

                data = {
                    "model": "llama3",  # TODO: Ability to change the model through yaml
                    "stream": True,
                    "prompt": prompt,
                }
                logger.debug(f"starting request on {self.messages=}")
                logger.debug("Performing request to LLM server...")

                # Perform the request and process the stream
                with requests.post(
                    self.completion_url,
                    json=data,
                    stream=True,
                ) as response:
                    sentence = []
                    for line in response.iter_lines():
                        if self.processing is False:
                            break  # If the stop flag is set from new voice input, halt processing
                        if line:  # Filter out empty keep-alive new lines
                            line = self._clean_raw_bytes(line)
                            logger.info(f"LLM response: {line['response']}")
                            next_token = self._process_line(line)
                            if next_token:
                                sentence.append(next_token)
                                # If there is a pause token, send the sentence to the TTS queue
                                if next_token in [
                                    ".",
                                    "!",
                                    "?",
                                    ":",
                                    ";",
                                    "?!",
                                    "\n",
                                    "\n\n",
                                ]:
                                    self._process_sentence(sentence)
                                    sentence = []
                    if self.processing:
                        if sentence:
                            self._process_sentence(sentence)
                    # Add end of stream token to the queue
                    self.tts_queue.put("<EOS>")
            except queue.Empty:
                time.sleep(PAUSE_TIME)

    def _process_sentence(self, current_sentence: List[str]):
        """
        Join text, remove inflections and actions, and send to the TTS queue.

        The LLM like to *whisper* things or (scream) things, and prompting is not a 100% fix.
        We use regular expressions to remove text between ** and () to clean up the text.
        Finally, we remove any non-alphanumeric characters/punctuation and send the text
        to the TTS queue.
        """
        sentence = "".join(current_sentence)
        sentence = re.sub(r"\*.*?\*|\(.*?\)", "", sentence)
        sentence = (
            sentence.replace("\n\n", ". ")
            .replace("\n", ". ")
            .replace("  ", " ")
            .replace(":", " ")
            .replace("*", " ")
        )
        logger.info(f"LLM sentence: {sentence}")
        if sentence:
            self.tts_queue.put(sentence)

    def _process_line(self, line):
        """
        Processes a single line of text from the LLM server.

        Args:
            line (dict): The line of text from the LLM server.
        """

        if not line["done"]:
            token = line["response"]
            return token
        return None

    def _clean_raw_bytes(self, line):
        """
        Cleans the raw bytes from the LLM server for processing.

        Coverts the bytes to a dictionary.

        Args:
            line (bytes): The raw bytes from the LLM server.
        """
        line = line.decode("utf-8")
        line = line.removeprefix("data: ")
        line = json.loads(line)
        return line


if __name__ == "__main__":
    vikky_config = VikkyConfig.from_yaml(VIKKY_CONFIG_PATH)
    logger.info("Vikky configurations loaded successfully")

    vikky = Vikky.from_config(vikky_config)

    vikky.start_listen_event_loop()
