from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

model_directory = "./models"  # Adjust this path if necessary

# Specify the model to use (e.g., GPT-2 or any other supported model)
model_name_or_path = "gpt2"  # Pre-trained GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Set `pad_token_id` to `eos_token_id` globally
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use `eos_token` as the `pad_token`
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for new token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to GPU/CPU


# Set the `pad_token_id` globally in the model's configuration
model.config.pad_token_id = tokenizer.pad_token_id

# Save the downloaded model locally for future use
os.makedirs(model_directory, exist_ok=True)
tokenizer.save_pretrained(model_directory)
model.save_pretrained(model_directory)
print(f"GPT-2 model saved to {model_directory}")