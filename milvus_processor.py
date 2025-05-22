"""
Milvus integration for the OSINT tool.
Handles processing content and inserting it into Milvus vector database.
"""
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Maximum size for text chunks to stay under Milvus VARCHAR limit (65535)
MAX_TEXT_SIZE = 65000

class MilvusProcessor:
    """
    A class to handle processing content and inserting it into Milvus vector database.
    """
    def __init__(self, milvus_ops, collection_name, embedding_model, batch_size, max_text_length):
        """
        Initialize the processor with necessary components.
        
        Args:
            milvus_ops: MilvusOperations instance
            collection_name: Name of the Milvus collection
            embedding_model: Model to generate embeddings
            batch_size: Number of records to insert in one batch
            max_text_length: Maximum length for text chunks
        """
        self.milvus_ops = milvus_ops
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_text_length = max_text_length
    
    def ensure_collection_exists(self, fields_config):
        """
        Ensure the Milvus collection exists, creating it if necessary.
        
        Args:
            fields_config: Configuration for collection fields
        """
        if not self.milvus_ops.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' does not exist. Creating it...")
            self.milvus_ops.create_collection(self.collection_name, fields_config)
            self.milvus_ops.create_index(self.collection_name)
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def process_content(self, urls, content_fetcher):
        """
        Process content from URLs and store in Milvus.
        
        Args:
            urls: List of URLs to process
            content_fetcher: ContentFetcher instance
        """
        overall_start_time = time.time()
        
        # Process URLs in parallel
        with ThreadPoolExecutor() as executor:
            list(tqdm(
                executor.map(
                    lambda url: self._process_single_url(url, content_fetcher),
                    urls
                ),
                total=len(urls),
                desc="Processing URLs"
            ))
        
        overall_end_time = time.time()
        print(f"Overall time taken to process URLs: {overall_end_time - overall_start_time:.2f} seconds")
    
    def _process_single_url(self, url, content_fetcher):
        """
        Process a single URL and insert its content into Milvus.
        
        Args:
            url: URL to process
            content_fetcher: ContentFetcher instance
        """
        print(f"Processing URL: {url}")
        url_start_time = time.time()
        
        try:
            # Fetch and preprocess content
            content = content_fetcher.fetch_from_url(url)
            if not content:
                print(f"Failed to fetch content from URL: {url}")
                return
            
            content = content_fetcher.preprocess_text(content)
            
            # Always chunk content to ensure it's below Milvus limits
            chunks = content_fetcher.split_into_chunks(content, min(self.max_text_length, MAX_TEXT_SIZE))
            
            if chunks:
                print(f"Split content into {len(chunks)} chunks.")
                self._process_chunks(chunks)
            else:
                print(f"No valid content chunks to process for URL: {url}")
                
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
        
        url_end_time = time.time()
        print(f"Time taken to process URL {url}: {url_end_time - url_start_time:.2f} seconds")
    
    def _process_chunks(self, chunks):
        """
        Process text chunks and prepare them for Milvus insertion.
        
        Args:
            chunks: List of text chunks
        """
        batch_data = []
        chunk_ids = {}  # Track IDs to prevent duplicates
        
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Skip empty chunks
            if not chunk or not chunk.strip():
                continue
                
            # If chunk exceeds size limit, split it into smaller chunks
            if len(chunk) > MAX_TEXT_SIZE:
                print(f"Chunk {chunk_idx} exceeds size limit, splitting into smaller chunks")
                sub_chunks = []
                for i in range(0, len(chunk), MAX_TEXT_SIZE):
                    sub_chunk = chunk[i:i + MAX_TEXT_SIZE]
                    if sub_chunk.strip():  # Only add non-empty sub-chunks
                        sub_chunks.append(sub_chunk)
                
                print(f"Split oversized chunk into {len(sub_chunks)} sub-chunks")
                
                # Process each sub-chunk
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    try:
                        # Generate embedding for sub-chunk
                        embedding = self.embedding_model.encode(sub_chunk)
                        
                        # Create a unique ID for this sub-chunk
                        sub_unique_id = (hash(sub_chunk) + sub_idx) % 2147483647
                        if sub_unique_id in chunk_ids:
                            sub_unique_id = (sub_unique_id + chunk_idx + sub_idx) % 2147483647
                        chunk_ids[sub_unique_id] = True
                        
                        # Add to batch
                        batch_data.append({
                            "id": sub_unique_id,
                            "embedding": embedding.tolist(),
                            "text": sub_chunk
                        })
                        
                        # Insert batch when batch size is reached
                        if len(batch_data) >= self.batch_size:
                            self._insert_batch(batch_data)
                            batch_data = []
                    except Exception as e:
                        print(f"Error processing sub-chunk {sub_idx} of chunk {chunk_idx}: {e}")
            else:
                try:
                    # Generate embedding for chunk
                    embedding = self.embedding_model.encode(chunk)
                    
                    # Create a unique ID for this chunk
                    unique_id = hash(chunk) % 2147483647  # Ensure ID is within INT32 range
                    if unique_id in chunk_ids:
                        unique_id = (unique_id + chunk_idx) % 2147483647
                    chunk_ids[unique_id] = True
                    
                    # Add to batch
                    batch_data.append({
                        "id": unique_id,
                        "embedding": embedding.tolist(),
                        "text": chunk
                    })
                    
                    # Insert batch when batch size is reached
                    if len(batch_data) >= self.batch_size:
                        self._insert_batch(batch_data)
                        batch_data = []
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
        
        # Insert any remaining data
        if batch_data:
            self._insert_batch(batch_data)
    
    def _insert_batch(self, data):
        """
        Insert a batch of data into Milvus.
        
        Args:
            data: List of data items to insert
        """
        if not data:
            return
            
        ids = []
        embeddings = []
        texts = []
          # Prepare data for insertion
        print(f"Preparing {len(data)} records for insertion into Milvus...")
        for item in tqdm(data, desc="Preparing data for Milvus"):
            # Check if text exceeds limit
            if len(item["text"]) > MAX_TEXT_SIZE:
                print(f"Warning: Found item with text exceeding MAX_TEXT_SIZE ({len(item['text'])} > {MAX_TEXT_SIZE})")
                # Try to truncate
                print(f"Truncating text to {MAX_TEXT_SIZE} characters")
                item["text"] = item["text"][:MAX_TEXT_SIZE]
                
            ids.append(item["id"])
            embeddings.append(item["embedding"])
            texts.append(item["text"])
        
        if not ids:
            print("No valid data to insert after filtering")
            return
            
        try:
            # Insert data
            print(f"Inserting {len(ids)} records into Milvus collection '{self.collection_name}'...")
            self.milvus_ops.insert_data(self.collection_name, ids, embeddings, texts)
            print(f"Successfully inserted {len(ids)} records into Milvus collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error inserting batch into Milvus: {e}")
    
    def search(self, query_text, top_k=10):
        """
        Search for relevant content in Milvus.
        
        Args:
            query_text: The query text
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        query_embedding = self.embedding_model.encode(query_text)
        
        search_results = self.milvus_ops.search_in_collection(
            self.collection_name, 
            query_embedding, 
            top_k=top_k
        )
        
        print(f"\nSearch results for query '{query_text}': {len(search_results)} matches")
        return search_results
