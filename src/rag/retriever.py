from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np
from src.database.mongo_client import MongoDBClient
from src.database.vector_store import MilvusClient
from src.rag.embeddings import create_embeddings
from src.rag.query_transformer import QueryTransformer

class HybridRetriever:
    def __init__(self, mongo_client: MongoDBClient, vector_store: MilvusClient, alpha: float = 0.6):
        self.mongo_client = mongo_client
        self.vector_store = vector_store
        self.query_transformer = QueryTransformer()
        self.alpha = alpha
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Chuẩn hóa điểm số về khoảng [0, 1]"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        return (scores - min_score) / (max_score - min_score + 1e-9)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Transform query if needed
        transformed_query = self.query_transformer.transform(query)
        
        # Get all documents first
        all_docs = list(self.mongo_client.collection.find({}))
        corpus = [doc['chunk_text'] for doc in all_docs]
        
        # Get semantic search scores
        query_embedding = create_embeddings([transformed_query])[0]
        semantic_results = self.vector_store.search(query_embedding, limit=len(corpus))
        
        # Extract and normalize semantic scores
        semantic_scores = np.zeros(len(corpus))
        for hit in semantic_results[0]:
            semantic_scores[hit.id] = hit.score
        semantic_scores = self._normalize_scores(semantic_scores)
        
        # Get and normalize BM25 scores
        bm25 = BM25Okapi([doc.split() for doc in corpus])
        bm25_scores = bm25.get_scores(transformed_query.split())
        bm25_scores = self._normalize_scores(bm25_scores)
        
        # Combine scores using alpha
        combined_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores
        
        # Get top_k results based on combined scores
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            doc = all_docs[idx]
            results.append({
                'text': doc['chunk_text'],
                'score': combined_scores[idx],
                'semantic_score': semantic_scores[idx],
                'bm25_score': bm25_scores[idx],
                'metadata': {
                    'years': doc['years'],
                    'event_types': doc['event_types'],
                    'tags': doc['tags']
                }
            })
        
        return results
