from typing import List, Dict
import json
import yaml
import argparse
import pandas as pd
#from src.database.mongo_client import MongoDBClient
from src.database.vector_store import MilvusClient
from src.data.data_processor import process_dataset


from data_processor import process_dataset
from pymongo import MongoClient
from typing import Dict, List
import yaml

class MongoDBClient:
    def __init__(self, config_path: str = r"config\config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['mongodb']
        
        self.client = MongoClient(host=config['host'], port=config['port'])
        self.db = self.client[config['database']]
        self.collection = self.db[config['collection']]

    def insert_documents(self, documents: List[Dict]):
        return self.collection.insert_many(documents)

    def search_by_metadata(self, query: Dict):
        return self.collection.find(query)


class DataLoader:
    def __init__(self, config_path: str = r'config\config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize database clients
        self.mongo_client = MongoDBClient(config_path)
        #self.vector_store = MilvusClient(config_path)

    def load_data(self, file_path: str) -> List[str]:
        data_df = pd.read_csv(file_path, delimiter="\t")
        return data_df['events'].values[:2]


    def process_and_store(self, data: List[str]) -> None:
        """Xử lý và lưu dữ liệu vào databases"""
        # Process data
        processed_data = process_dataset(
            data,
            chunk_size=self.config['model']['chunk_size'],
            model_name=self.config['model']['embedding_model']
        )
        # Prepare documents for MongoDB
        documents = []
        for i, (chunk, metadata) in enumerate(zip(processed_data['chunks'], processed_data['metadata'])):
            doc = {
                'original_index': i,
                'chunk_text': chunk,
                **metadata
            }
            documents.append(doc)
        
        # Store in MongoDB
        self.mongo_client.insert_documents(documents)
        
        # Store in Milvus
        # self.vector_store.insert_embeddings(
        #     ids=processed_data['original_indices'],
        #     embeddings=processed_data['embeddings']
        # )
        print(f"Successfully processed and stored {len(documents)} documents")

def load_and_store_data(file_path: str) -> None:
    """
    Hàm tiện ích để load và store dữ liệu
    Args:
        file_path (str): Đường dẫn đến tệp dữ liệu
    """
    print(f"Loading data from {file_path}")
    loader = DataLoader()
    
    data = loader.load_data(file_path)
    #print(data)
    print(f"Loaded {len(data)} records. Processing and storing...")
    loader.process_and_store(data)
    print("Data loading process completed!")

if __name__ == "__main__":
    file = r'D:\mini project\NLP\RAG\history_rag\data\raw\data.csv'
    
    try:
        load_and_store_data(file)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
