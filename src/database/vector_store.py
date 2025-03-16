from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
import yaml
from typing import List


class MilvusClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['milvus']
        
        connections.connect(host=config['host'], port=config['port'])
        self.collection_name = config['collection']
        self._create_collection()

    def _create_collection(self):
        if self.collection_name not in Collection.list_collections():
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            schema = CollectionSchema(fields=fields)
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index(field_name="embedding", index_type="IVF_FLAT")
        else:
            self.collection = Collection(self.collection_name)

    def insert_embeddings(self, ids: List[int], embeddings: List[List[float]]):
        entities = [ids, embeddings]
        self.collection.insert(entities)

    def search(self, query_embedding: List[float], top_k: int = 5):
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )
        return results