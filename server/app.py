import logging
from flask import Flask, jsonify, request, Response, stream_with_context
import requests
import json

from urllib3 import HTTPConnectionPool

app = Flask(__name__)
ollamaAPI = "http://localhost:11434/api"
model = "llama3"
CHUNK_SIZE = 4096


@app.route("/", methods=["GET"])
def index():
    return "Hello, my name is Vikky. I am your personal assistant and I am here to help you with your queries."


@app.route("/generate", methods=["POST"])
def ollama():
    try:
        response = requests.post(ollamaAPI + "/generate", json=request.json)

        return Response(
            stream_with_context(response.iter_content(chunk_size=CHUNK_SIZE)),
            content_type=response.headers["content-type"],
        )
    except HTTPConnectionPool as e:
        return jsonify({"error": "Error connecting to ollama server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def ollama_chat():
    try:
        messages = request.json["messages"]
        data = {"model": model, "messages": messages}
        response = requests.post(ollamaAPI + "/chat", json=data)
        return Response(
            stream_with_context(response.iter_content(chunk_size=CHUNK_SIZE)),
            content_type=response.headers["content-type"],
        )
    except HTTPConnectionPool as e:
        return jsonify({"error": "Error connecting to ollama server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def temporary():
    def textResponse():
        try:
            prompt = request.json["prompt"]
            data = {"model": model, "prompt": prompt, "stream": True}
            response = requests.post(ollamaAPI, json=data, stream=True)
            full_response = ""

            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                app.logger.info("Chunk: %s", chunk)
                # Decode the line from bytes to string
                line = chunk.decode("utf-8")
                app.logger.info("Line Converted: %s", line)
                # Parse the JSON data in each line
                data = json.loads(line)
                app.logger.info("Data: %s", data)

                # Append the response to the full response
                full_response += data.get("response", "")

                if data.get("done", False):
                    app.logger.info("Done")
                    break
            # Return the full response to the end user
            return jsonify({"response": full_response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, host="0.0.0.0", port=7777)
