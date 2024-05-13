## Use Case

Vikky is a personal assistant that helps developers to be more productive and efficient. Vikky is a bot that can be integrated with the developer's workspace and can be used to automate tasks, provide reminders, and help developers to stay focused. It is a physical bot that can be placed on the developer's desk and can be controlled using voice commands. Raspberry PI is used as the hardware platform for Vikky. Vikky's main focus is to help developers to be more productive and efficient by providing reminders, automating tasks, and helping developers to stay focused.

## Features

1. Vikky can be used to automate tasks like sending emails, setting reminders, and fetching information from the internet.
2. Vikky can be used to provide reminders to the developer.
3. Vikky can be used to help developers to stay focused by blocking distracting websites.
4. Vikky can be used to control the developer's workspace using voice commands.
5. Vikky can perform self research and provide information to the developer by automatically creating a summary and store in NAS
6. Vikky is there to listen to the developer's thoughts while debugging and provide suggestions.
7. Vikky has camera, sensors and can be used to monitor the developer's workspace.
   - IOT solutions
     - Monitor the developer's workspace
     - Monitor the developer's health
     - Monitor the developer's mood
     - Monitor the developer's productivity
     - Monitor the developer's posture
     - Monitor the developer's screen time
     - Monitor the developer's water intake
     - Monitor the developer's sleep
8. Automating tasks
   - Automate repetitive tasks
   - Automate the developer's workflow
   - Automate the developer's workspace
   - Automate the developer's life
9. Listens into meetings and provides a summary

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
         1. Linux with [CUDA](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#nvidia-gpu-support): `make libwhisper.so -j`
         2. Mac with [CoreML](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file#core-ml-support): `make -j`
5. For the LLM, you have two option:
   1. Use a commercial API or install an inference backend yourself, such as Ollama:
      1. Find and install a backend with an OpenAI compatible API (most of them)
      2. Edit the glados_config.yaml
         1. update `completion_url` to the URL of your local server
         2. for commercial APIs, add the `api_key`
         3. remove the LlamaServer configurations (make them null)
