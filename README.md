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



## Run The Frontend 
### **Solution: Install or Fix Node.js Installation**

1. **Check Node.js Installation**:
   - Run the following command to check if Node.js is installed:
   
     ```bash
     node -v
     ```
   - If Node.js is installed, this command will print the version number, e.g., `v18.16.1`.
   - If not, proceed to install Node.js.

2. **Install Node.js**:
   - Visit the official Node.js website: [Node.js Download](https://nodejs.org/)
   - Download the **LTS version** (recommended for most users).
   - Install Node.js by running the downloaded installer.
     - During installation, ensure that the **"Add to PATH"** option is selected so the `node` and `npx` commands will be available from the terminal.

3. **Verify Installation**:
   - After installation, verify that both Node.js and npm (Node Package Manager) are properly installed:
     
     ```bash
     node -v
     npm -v
     ```
   - The `node -v` command should output the installed Node.js version.
   - The `npm -v` command should output the npm version.

4. **Try Running `npx` Again**:
   - Once Node.js is installed, `npx` should also work because it is included with npm.
   - Run the command to create your Next.js app:
     
     ```bash
     npx create-next-app@latest .
     ```

---

### **If `npx` Still Doesn't Work**

If `npx` is still not recognized after installing Node.js, it may be due to an issue with the PATH environment variable.

#### **Fix the PATH Environment Variable (Windows)**:
1. Press **Win + S** and search for **Environment Variables**.
2. Click on **Edit the system environment variables**.
3. In the **System Properties** window, click on the **Environment Variables** button.
4. Under **System Variables**, look for the `Path` variable and select it, then click **Edit**.
5. Add the following path (replace `C:\Program Files\nodejs` with the actual installation path of Node.js):
   
   ```plaintext
   C:\Program Files\nodejs
   ```
6. Click **OK** to save changes and restart your terminal.

---

### **Alternative: Use npm to Create the App**

If `npx` still doesn't work, you can use `npm` directly to create a Next.js app:

```bash
npm init next-app@latest .
```

This command achieves the same result as `npx create-next-app@latest`.

---

### **Verify the Frontend Setup**

Once the command runs successfully, follow the prompts to set up your Next.js app. After installation:

1. Navigate to the project directory:
   
   ```bash
   cd frontend
   ```

2. Start the development server:
   
   ```bash
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:3000`.