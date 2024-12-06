import os
import requests
from bs4 import BeautifulSoup

def crawl_web_directory(url, output_dir):
    """
    Crawl the given web directory and save HTML content.

    Args:
        url (str): The URL of the web directory.
        output_dir (str): Directory to save the crawled data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return
    
    soup = BeautifulSoup(response.content, "html.parser")
    # Extract links or text depending on the structure of the web directory
    links = soup.find_all("a")
    
    for link in links:
        href = link.get("href")
        if href and href.startswith("http"):
            try:
                page_response = requests.get(href)
                if page_response.status_code == 200:
                    file_name = os.path.join(output_dir, f"{href.split('/')[-1]}.html")
                    with open(file_name, "w", encoding="utf-8") as file:
                        file.write(page_response.text)
            except Exception as e:
                print(f"Error fetching {href}: {e}")

if __name__ == "__main__":
    web_directory_url = "https://example.com/documents"
    output_directory = "../data/raw"
    crawl_web_directory(web_directory_url, output_directory)