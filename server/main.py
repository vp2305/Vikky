import requests
import json
import pyttsx3


# Start the audio stream
def speakStream(full_response):
    engine = pyttsx3.init()
    engine.say("The meaning of life is 42")
    engine.runAndWait()


def main():
    url = "http://localhost:7777/generate"
    data = {"prompt": "What is the meaning of life?"}

    # Retrieve the response from the server using POST method the result is in stream
    response = requests.post(url, json=data, stream=True)
    full_response = ""

    # Iterate over the response content
    for chunk in response.iter_content(chunk_size=4096):
        # Decode the line from bytes to string
        line = chunk.decode("utf-8")
        # Parse the JSON data in each line
        data = json.loads(line)

        # Remove the unnecessary character such as *#
        if data.get("response", ""):
            data["response"] = (
                data["response"].replace("*", "").replace("#", "").replace('"', "")
            )

        # Append the response to the full response
        full_response += data.get("response", "")

    speakStream(full_response)


main()
