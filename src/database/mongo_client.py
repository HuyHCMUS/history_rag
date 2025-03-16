from pymongo import MongoClient
from typing import Dict, List
import yaml

class MongoDBClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['mongodb']
        
        self.client = MongoClient(host=config['host'], port=config['port'])
        self.db = self.client[config['database']]
        self.collection = self.db[config['collection']]

    def insert_documents(self, documents: List[Dict]):
        return self.collection.insert_many(documents)

    def search_by_metadata(self, query: Dict):
        return self.collection.find(query)