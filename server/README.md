# USAGE
- docker build -t vikky . --add-host=host.docker.internal:host-gateway
- docker run --name="Vikky" -p 7777:7777 vikky
- Start the ollama server
 - Navigate to the docker desktop
 - ollama serve
- Install llama3 using another terminal
  - docker exec -it Vikky ollama pull llama3


# IP address that I can invoke the server with
- http://172.16.0.15:7777


# Test commands that may run
- docker run --name="Vikky" --network="host" vikky

# Check if the ollama is running
- curl -X GET http://127.0.0.1:11434