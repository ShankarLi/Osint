from pymilvus import connections, Collection, CollectionSchema, FieldSchema, utility, db
from pymilvus.orm.types import DataType
import time
import json
import tempfile

class MilvusOperations:
    def __init__(self, host, port, timeout, db_name):
        """
        Initialize the MilvusOperations class with host, port, timeout, and database name.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.db_name = db_name

    def connect_to_milvus(self):
        """
        Connect to the Milvus server.
        """
        try:
            connections.connect(alias="default", host=self.host, port=self.port, timeout=self.timeout, db_name=self.db_name)
            print("Connected to Milvus")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def has_collection(self, collection_name):
        """
        Check if a collection with the given name exists in Milvus.
        """
        return utility.has_collection(collection_name)

    def create_collection(self, collection_name, fields_config):
        """
        Create a collection in Milvus with the specified name and fields configuration.
        The fields configuration is passed as a list of dictionaries.
        """
        fields = []
        for field in fields_config:
            if field['dtype'] == "FLOAT_VECTOR":
                fields.append(FieldSchema(name=field['name'], dtype=DataType[field['dtype']], dim=field['dim']))
            elif field['dtype'] == "VARCHAR":
                fields.append(FieldSchema(name=field['name'], dtype=DataType[field['dtype']], max_length=field['max_length']))
            else:
                fields.append(FieldSchema(name=field['name'], dtype=DataType[field['dtype']], is_primary=field.get('is_primary', False)))

        schema = CollectionSchema(fields=fields, description="Collection for storing embeddings and text")
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created.")
        return collection
    
    def create_index(self, collection_name):
        """
        Create indexes for the embedding and text fields in the specified Milvus collection.
        """
        collection = Collection(name=collection_name)

        # Check if an index already exists for the embedding field
        try:
            embedding_index_info = collection.indexes
            if any(index.field_name == "embedding" for index in embedding_index_info):
                print(f"Index for 'embedding' field already exists in collection '{collection_name}'. Skipping creation.")
            else:
                embedding_index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 512}
                }
                print(f"Creating index for 'embedding' field in collection '{collection_name}'...")
                collection.create_index(field_name="embedding", index_params=embedding_index_params)
                print("Index for 'embedding' field created.")
        except Exception as e:
            print(f"Error checking or creating index for 'embedding' field: {e}")

        # # Create index for the text field
        # text_index_params = {
        #     "index_type": "SPARSE_INVERTED_INDEX",
        #     "metric_type": "BM25"
        # }
        # print(f"Creating index for 'text' field in collection '{collection_name}'...")
        # collection.create_index(field_name="text", index_params=text_index_params)
        # print(f"Index for 'text' field created.")

    def insert_data(self, collection_name, ids, embeddings, texts):
        """
        Insert data into the specified Milvus collection, including text values.
        """
        collection = Collection(name=collection_name)
        data = [ids, embeddings, texts]
        collection.insert(data)
        print(f"Inserted {len(ids)} records into collection '{collection_name}'.")
    
    def do_bulk_insert(self, collection_name, data, column_names, partition_name=None, timeout=None, using="default"):
        """
        Perform bulk insertion of data into the specified Milvus collection.
        Automatically formats the data into the required JSON structure.

        Parameters:
        - collection_name (str): The name of the target collection.
        - data (list[dict]): The data to be inserted.
        - column_names (list[str]): The column names corresponding to the data.
        - partition_name (str, optional): The name of the partition to insert data into.
        - timeout (float, optional): Timeout duration for the operation.
        - using (str, optional): The alias of the employed connection.

        Returns:
        - int: A bulk-insert task ID.
        """
        try:
            # Generate a unique partition name
            if partition_name is None:
                partition_name = f"partition_{int(time.time())}"            # Create the partition
            collection = Collection(name=collection_name)
            try:
                if not collection.has_partition(partition_name):
                    collection.create_partition(partition_name)
                    print(f"Created new partition: {partition_name}")
                else:
                    print(f"Partition {partition_name} already exists. Proceeding with insertion.")
            except Exception as e:
                print(f"Error handling partition: {e}")
                if "already exists" not in str(e).lower():
                    raise

            # Format data into the required JSON structure
            formatted_data = {"rows": [
                {column: item[column] for column in column_names} for item in data
            ]}
            # print(f"Formatted data for bulk insert: {formatted_data}")

            # Save formatted data to a temporary JSON file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_file:
                json.dump(formatted_data, temp_file)
                temp_file_path = temp_file.name
                print(f"Temporary file created: {temp_file_path}")

            # Perform bulk insert
            task_id = utility.do_bulk_insert(
                collection_name=collection_name,
                files=[temp_file_path],
                partition_name=partition_name,
                timeout=timeout
                # using=using
            )
            print(f"Bulk insert initiated with task ID: {task_id} into partition: {partition_name}")
            return task_id
        except Exception as e:
            print(f"Failed to perform bulk insert: {e}")
            raise

    def search_in_collection(self, collection_name, query_embedding, top_k=10):
        """
        Search for similar embeddings in the specified Milvus collection.
        """
        collection = Collection(name=collection_name)
        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        # print(f"search_in_collection results : {results}")
        # Extract the text values from the results
        extracted_texts = []
        for result in results:
            for hit in result:
                text_value = hit.entity.get("text")
                if text_value:
                    extracted_texts.append(text_value)

        # print(f"Extracted texts: {extracted_texts}")
        return extracted_texts
    
    def drop_collection(self, collection_name):
        """
        Drop the specified Milvus collection.
        """
        if self.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' has been dropped.")
        else:
            print(f"Collection '{collection_name}' does not exist. No action taken.")

    def disconnect_from_milvus(self):
        """
        Disconnect from the Milvus server.
        """
        connections.disconnect(alias="default")
        print("Disconnected from Milvus")
