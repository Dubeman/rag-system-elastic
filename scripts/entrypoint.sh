#!/bin/bash
set -e

# Start Ollama server in the background
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
sleep 5

# Pull the model
ollama pull "$MODEL_NAME"

# Wait for Ollama server to exit (or bring to foreground)
wait $OLLAMA_PID
