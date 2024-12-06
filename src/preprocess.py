import os
from bs4 import BeautifulSoup

def preprocess_data(input_dir, output_dir):
    """
    Preprocess raw HTML files by extracting text content.

    Args:
        input_dir (str): Directory containing raw HTML files.
        output_dir (str): Directory to save preprocessed text files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.endswith(".html"):
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
                # Extract relevant text (customize based on structure)
                text = soup.get_text()
                clean_text = " ".join(text.split())  # Remove excess whitespace
                
                output_file = os.path.join(output_dir, f"{filename}.txt")
                with open(output_file, "w", encoding="utf-8") as output:
                    output.write(clean_text)

if __name__ == "__main__":
    raw_data_dir = "../data/raw"
    processed_data_dir = "../data/processed"
    preprocess_data(raw_data_dir, processed_data_dir)