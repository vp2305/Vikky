## Use Case

Vikky is a personal assistant that helps developers to be more productive and efficient. Vikky is a bot that can be integrated with the developer's workspace and can be used to automate tasks, provide reminders, and help developers to stay focused. It is a physical bot that can be placed on the developer's desk and can be controlled using voice commands. Raspberry PI is used as the hardware platform for Vikky. Vikky's main focus is to help developers to be more productive and efficient by providing reminders, automating tasks, and helping developers to stay focused.

## Regular installation

If you want to install the TTS Engine on your machine, please follow the steps
below.  This has only been tested on Linux, but I think it will work on Windows with small tweaks.
If you are on Windows, I would recommend WSL with an Ubuntu image.  Proper Windows and Mac support is in development.

1. Install the [`espeak`](https://github.com/espeak-ng/espeak-ng) synthesizer
   according to the [installation
   instructions](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)
   for your operating system.
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt` on Mac or Linux systems without an Nvidia GPU, and `pip install -r
   requirements_cuda.txt` if you have a modern Nvidia GPU.
3.  Download the models:
    1.  [voice recognition model](https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin?download=true)
4. For voice recognition, we use [Whisper.cpp](https://github.com/ggerganov/whisper.cpp)
   1. Compile them yourself. 
      1. To pull the code, from the Vikky directory use: `git submodule update --init --recursive`
      2. Move to the right subdirectory: `cd submodules/whisper.cpp`
      3. Compile for your system [(see the Documentation)](https://github.com/ggerganov/whisper.cpp), e.g.
         1. Raspberry PI with: `make libwhisper.so -j` or `make -j`
            1. sudo apt install ffmpeg
            2. No such file or directory as SDL.h => sudo apt install libsdl2-dev
         3. Mac with [CoreML](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#core-ml-support): `WHISPER_COREML=1 make -j`
5. For the LLM, you have two option:
   1. Use a commercial API or install an inference backend yourself, such as Ollama:
      1. Find and install a backend with an OpenAI compatible API (most of them)
      2. Edit the glados_config.yaml
         1. update `completion_url` to the URL of your local server
         2. for commercial APIs, add the `api_key`
         3. remove the LlamaServer configurations (make them null)
