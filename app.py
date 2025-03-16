import streamlit as st
import os
from src.rag.retriever import HybridRetriever
from src.database.mongo_client import MongoDBClient
from src.database.vector_store import MilvusClient
from src.llm.chain import HistoryQAChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
@st.cache_resource
def init_components():
    mongo_client = MongoDBClient()
    vector_store = MilvusClient()
    retriever = HybridRetriever(mongo_client, vector_store)
    qa_chain = HistoryQAChain()
    return retriever, qa_chain

def main():
    st.title("🏯 Hỏi đáp Lịch sử Việt Nam")
    
    # Initialize components
    retriever, qa_chain = init_components()
    
    # User input
    question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Ai là vị vua đầu tiên của nhà Lý?")
    
    if st.button("Tìm câu trả lời"):
        if not question:
            st.warning("Vui lòng nhập câu hỏi!")
            return
            
        with st.spinner("Đang tìm câu trả lời..."):
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(question, top_k=3)
            
            # Generate answer
            answer = qa_chain.answer(question, retrieved_docs)
            
            # Display answer
            st.write("### Câu trả lời:")
            st.write(answer)
            
            # Display sources
            with st.expander("Xem nguồn tham khảo"):
                for i, doc in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Nguồn {i}** (Độ tương đồng: {doc['score']:.2f})")
                    st.write(doc['text'])
                    st.write("---")

if __name__ == "__main__":
    main()