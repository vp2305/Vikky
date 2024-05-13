FROM ubuntu:20.04

# Set the working directory in the container
WORKDIR /vikky

# Copy the content of the local src directory to the working directory
COPY . /vikky

# Install python 3.12.3
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Install ollama onto the container using curl 
RUN apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

# Make port 5000 available to the world outside this container
EXPOSE 7777
CMD python3 app.py