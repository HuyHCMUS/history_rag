from src.data.data_processor import process_dataset
from src.database.mongo_client import MongoDBClient
from src.database.vector_store import MilvusClient
from src.rag.retriever import HybridRetriever
import yaml

def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    
    # Initialize database clients
    mongo_client = MongoDBClient()
    vector_store = MilvusClient()
    
    # Load and process data
    with open("data/raw/history_events.txt", 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    processed_data = process_dataset(
        data,
        chunk_size=config['model']['chunk_size'],
        model_name=config['model']['embedding_model']
    )
    
    # Store data in MongoDB
    documents = []
    for i, (chunk, metadata) in enumerate(zip(processed_data['chunks'], processed_data['metadata'])):
        doc = {
            'original_index': i,
            'chunk_text': chunk,
            **metadata
        }
        documents.append(doc)
    
    mongo_client.insert_documents(documents)
    
    # Store embeddings in Milvus
    vector_store.insert_embeddings(
        ids=processed_data['original_indices'],
        embeddings=processed_data['embeddings']
    )
    
    # Initialize retriever
    retriever = HybridRetriever(mongo_client, vector_store)
    
    # Example usage
    query = "Ai là vị vua đầu tiên của nhà Lý?"
    results = retriever.retrieve(query, top_k=5)
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Score: {result['score']}")
        print(f"Year: {result['metadata']['years']}")
        print(f"Event types: {result['metadata']['event_types']}")
        print(f"Tags: {result['metadata']['tags']}")
        print("---")

if __name__ == "__main__":
    main()