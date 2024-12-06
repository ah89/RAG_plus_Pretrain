from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

def train_language_model(data_dir, model_name="gpt2", output_dir="../models"):
    """
    Fine-tune a language model on custom data.

    Args:
        data_dir (str): Directory containing processed text files.
        model_name (str): Pre-trained model name.
        output_dir (str): Directory to save the fine-tuned model.
    """
    # Load processed text files into a dataset
    dataset = load_dataset("text", data_files={"train": f"{data_dir}/*.txt"})
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    # Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )
    
    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    processed_data_dir = "../data/processed"
    train_language_model(processed_data_dir)