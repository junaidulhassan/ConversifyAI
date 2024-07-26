import requests
from bs4 import BeautifulSoup
import re


class Scraper:
    def __init__(self, url):
        self.url = url
    
    def __write_txt_file(self, text, string):
        # Define the file path
        file_path = "/media/junaid-ul-hassan/248ac48e-ccd4-4707-a28b-33cb7a46e6dc/LLMs Projects/Web_pilot/Scraped_Text/text_file.txt"
        line_length = 20
        
        # Format the text
        words = text.split()
        formatted_text = re.sub(r'\s+', ' ', string)

        # Open the file in write mode to clear its content and write new data
        with open(file_path, 'w') as file:
            for i in range(0, len(words), line_length):
                file.write(' '.join(words[i:i + line_length]) + '\n')
    
    def scrape_website(self):
        # Define the headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
        }

        # Send a GET request to the URL with headers
        response = requests.get(self.url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code != 200:
            return f"Failed to retrieve the webpage. Status code: {response.status_code}"

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Function to extract and clean text from a list of elements
        def extract_text(elements):
            return [element.get_text(separator=" ").strip() for element in elements if element.get_text().strip()]

        # Extract headings, subheadings, paragraphs, and list items
        headings = extract_text(soup.find_all('h1'))
        paragraphs = extract_text(soup.find_all('p'))
        list_items = extract_text(soup.find_all('li'))
        articles = extract_text(soup.find_all('article'))
        sections = extract_text(soup.find_all('section'))
        block_quote = extract_text(soup.find_all('blockquote'))



        # Combine all extracted text
        all_text = headings+paragraphs+articles+block_quote+sections+list_items
        
        # Join the lines back together without empty lines
        result_string = ' '.join(all_text)

        # Clean up any extra whitespace
        formatted_text = re.sub(r'\s+', ' ', result_string)
        
        self.__write_txt_file(
            text=formatted_text,
            string = result_string
        )

        return result_string, formatted_text
    
    

