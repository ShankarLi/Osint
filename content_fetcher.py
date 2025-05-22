"""
Web content fetching and processing for the OSINT tool.
Handles fetching content from URLs and preprocessing text.
"""
import requests
from bs4 import BeautifulSoup

# Maximum size for text chunks to stay under Milvus VARCHAR limit (65535)
MAX_TEXT_SIZE = 65000

class ContentFetcher:
    """Handles fetching and processing content from web sources."""
    
    @staticmethod
    def fetch_from_url(url, timeout=10):
        """
        Fetch and parse content from a given URL.
        
        Args:
            url: The URL to fetch content from
            timeout: Request timeout in seconds
            
        Returns:
            Extracted text content or None if fetch fails
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
    
    @staticmethod
    def preprocess_text(text):
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove HTML tags if any remain
        text = BeautifulSoup(text, 'html.parser').get_text()
        # Remove leading/trailing whitespace
        text = text.strip()
        return text    
    
    @staticmethod
    def split_into_chunks(text, max_length):
        """
        Split text into smaller chunks of a specified maximum length.
        
        Args:
            text: Text to split
            max_length: Maximum length of each chunk
            
        Returns:
            List of text chunks
        """
        # Ensure max_length doesn't exceed Milvus VARCHAR limit
        safe_max_length = min(max_length, MAX_TEXT_SIZE)
        
        if len(text) <= safe_max_length:
            return [text]
            
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_length + word_len > safe_max_length:
                if current_chunk:  # Only add if there's content
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Final validation to ensure no chunk exceeds the limit
        valid_chunks = []
        for chunk in chunks:
            if len(chunk) > safe_max_length:
                # If a single chunk is still too large, split it at character level
                for i in range(0, len(chunk), safe_max_length):
                    sub_chunk = chunk[i:i + safe_max_length]
                    if sub_chunk:
                        valid_chunks.append(sub_chunk)
            else:
                valid_chunks.append(chunk)
                
        return valid_chunks
