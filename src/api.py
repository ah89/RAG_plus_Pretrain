from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import os
import torch
import faiss
import numpy as np
from datasets import Dataset



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow specific origins (frontend URL)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Path to the pre-trained model directory
model_directory = "/app/models"  # Adjust this path if necessary

# Check if the model directory exists and contains the required files
if os.path.exists(model_directory) and os.path.isfile(os.path.join(model_directory, "pytorch_model.bin")):
    print(f"Loading pre-trained model from {model_directory}")
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory)
else:
    print("Pre-trained model not found. Loading GPT-2 from Hugging Face and saving it locally...")
    model_name_or_path = "gpt2"  # Pre-trained GPT-2
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Set `pad_token_id` to `eos_token_id` globally
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use `eos_token` as the `pad_token`
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for new token

    # Set the `pad_token_id` globally in the model's configuration
    model.config.pad_token_id = tokenizer.pad_token_id

    # Save the downloaded model locally for future use
    os.makedirs(model_directory, exist_ok=True)
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)
    print(f"GPT-2 model saved to {model_directory}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Directory for temporary storage
TEMP_DIR = "/app/temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Initialize the embedding model for the retriever
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight transformer for embeddings

# Create or load FAISS index
knowledge_base = []  # Stores raw text for debugging/retrieval
if os.path.exists("knowledge_base.index"):
    index = faiss.read_index("knowledge_base.index")
    print("FAISS index loaded successfully.")
else:
    print("No FAISS index found. Initializing a new index.")
    index = faiss.IndexFlatL2(384)  # Use L2 distance (384 = embedding dimension for MiniLM)
    print("Initialized a new FAISS index.")

# --- ENDPOINT: Chat with RAG ---
@app.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint with Retrieval-Augmented Generation (RAG).
    """
    data = await request.json()
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    
    # Step 1: Retrieve relevant context using FAISS
    query_embedding = embedding_model.encode([query])  # Encode the query into an embedding
    distances, indices = index.search(np.array(query_embedding), k=2)  # Retrieve top-2 similar docs


    combined_input = ""
    if indices[0][0] == -1:
        # No result found and\or KB is empty
        combined_input = f"Query: {query}"
    else:
        # Combine the top-k retrieved knowledge
        retrieved_contexts = " ".join([knowledge_base[i] for i in indices[0]])

        # Step 2: Combine the query with the retrieved context
        combined_input = f"Context: {retrieved_contexts}\nQuery: {query}"

    # Step 3: Tokenize the input and move it to the GPU/CPU
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Step 4: Generate a response using the fine-tuned model
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=300,
        max_length=300,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        top_p=0.9,
        temperature=0.7
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}

# --- ENDPOINT: Pretrain the model ---
@app.post("/pretrain")
async def pretrain(request: Request):
    """
    Pretrain endpoint: Crawls a web page, preprocesses the content, and fine-tunes the model.
    """
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    # Step 1: Crawl the webpage
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch {url}. Status code: {response.status_code}")

        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()  # Extract all text from the page
        clean_text = " ".join(text.split())  # Remove excess whitespace
        
        
        # Split the text into smaller chunks for fine-tuning
        chunks = []
        max_chunk_size = 512
        chunk_size = min(max(clean_text[0:max_chunk_size].rfind(' '),clean_text[0:max_chunk_size].rfind('.')) , max_chunk_size)  # Max tokens for first chunk
        for i in range(0, len(clean_text), chunk_size):
            chunks.append(clean_text[i:i + chunk_size] )
            chunk_size = min(max(clean_text[i: i + max_chunk_size].rfind(' '),clean_text[i: i + max_chunk_size].rfind('.')), max_chunk_size)  # Max tokens for next chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error crawling the webpage: {str(e)}")

    # Step 2: Fine-tune the model on the crawled data
    try:
        # Add a pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Add a custom [PAD] token
            model.resize_token_embeddings(len(tokenizer))  # Resize the model's embeddings to include the new token

        # Load crawled data into a Dataset
        dataset = Dataset.from_dict({"text": chunks})

        # Tokenize the dataset
        def tokenize_function(examples):
            inputs = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
            inputs["labels"] = inputs["input_ids"].copy()  # Set labels equal to input_ids for causal LM
            return inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Prepare the dataset for training by renaming keys
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_directory,
            eval_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            num_train_epochs=3,  # Train for multiple epochs
            save_strategy="no",
            logging_dir="./logs",
        )

        # Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Save the updated model
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fine-tuning the model: {str(e)}")

    return {"message": "Model fine-tuned successfully on the crawled data."}

# --- ENDPOINT: Add data to the Knowledge Graph ---
@app.post("/add_to_kg")
async def add_to_kg(request: Request):
    """
    Add data to the Knowledge Graph (KG) by crawling a URL, processing the text, 
    and indexing it into the retrieval system.
    """
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    # Step 1: Crawl the webpage
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch {url}. Status code: {response.status_code}")

        # Extract and clean the text
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()  # Extract all text from the page
        clean_text = " ".join(text.split())  # Clean excessive whitespace

        # Split the text into chunks (for embedding purposes)
        # Split the text into smaller chunks for fine-tuning
        chunks = []
        max_chunk_size = 512
        chunk_size = min(max(clean_text[0:max_chunk_size].rfind(' '),clean_text[0:max_chunk_size].rfind('.')) , max_chunk_size)  # Max tokens for first chunk
        for i in range(0, len(clean_text), chunk_size):
            chunks.append(clean_text[i:i + chunk_size] )
            chunk_size = min(max(clean_text[i: i + max_chunk_size].rfind(' '),clean_text[i: i + max_chunk_size].rfind('.')), max_chunk_size)  # Max tokens for next chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error crawling the webpage: {str(e)}")

    # Step 2: Convert text chunks into embeddings
    try:
        # Encode chunks into embeddings using the SentenceTransformer
        chunk_embeddings = np.array(embedding_model.encode(chunks))

        # Add the embeddings to the FAISS index
        index.add(chunk_embeddings)

        # Optionally, store the raw text chunks alongside their embeddings for debugging
        global knowledge_base  # Extend the existing knowledge base
        knowledge_base.extend(chunks)

        # Save the updated FAISS index for persistence
        faiss.write_index(index, "knowledge_base.index")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing data to KG: {str(e)}")

    return {"message": f"Successfully added {len(chunks)} chunks to the Knowledge Graph from {url}."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)