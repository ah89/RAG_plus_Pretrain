# RAG and Pretrain Combined

Loop-in RAG and Pretrain as chat agent to the LLM structure. A chatbot that consumes data from web directories, trains a language model, and deploys as a Docker container.

## **Iinitialization**

1. Run it:

   ```bash
   python save_model.py
   ```

2. This will create a `models` directory in your current working directory with the following files:

   ```
   models/
   ├── config.json
   ├── merges.txt
   ├── vocab.json
   ├── special_tokens_map.json
   ├── tokenizer_config.json
   ├── pytorch_model.bin
   ```

## **Run The API**
### **Run with Docker**
1. Build the Docker image:

   ```bash
   docker-compose up --build
   ```

or

   ```bash
   docker build -t chatbot-app .
   ```

2. Run the container:

   ```bash
   docker run --gpus all obs_llm-chatbot
   ```

### **1. Testing the Chatbot**

1. Make sure the `models` directory contains your fine-tuned model files.
2. Rebuild the Docker image with the updated `Dockerfile`.
3. Run the container and test the chatbot API by sending a POST request to the `/chat` endpoint.

Example (using `curl`):

```bash
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{"query": "Hello, how can I help you?"}'
```

If everything is set up correctly, the chatbot should return a valid response.




### **2. Call the `/pretrain` Endpoint**
Send a POST request to the `/pretrain` endpoint with the webpage URL to crawl and fine-tune the model.

Example using `curl`:

```bash
curl -X POST "http://localhost:8000/pretrain" \
-H "Content-Type: application/json" \
-d '{"url": "https://example.com/some_page"}'
```

Expected response:

```json
{
  "message": "Model fine-tuned successfully on the crawled data."
}
```
