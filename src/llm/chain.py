from langchain_core.prompts import PromptTemplate
from langchain_core.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
from typing import List, Dict

class HistoryQAChain:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.llm = GoogleGenerativeAI(
            temperature=0.7,
            model_name=model_name,
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Bạn là một chuyên gia về lịch sử Việt Nam. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
            Nếu không thể trả lời từ ngữ cảnh, hãy nói rằng bạn không có đủ thông tin.

            Ngữ cảnh:
            {context}

            Câu hỏi: {question}

            Trả lời chi tiết và chính xác. Nếu có năm cụ thể trong ngữ cảnh, hãy đề cập đến năm đó.
            Trả lời:"""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format các document thành một đoạn văn bản context"""
        context_parts = []
        for doc in retrieved_docs:
            text = doc['text']
            score = doc['score']
            if score > 0.5:  # Chỉ sử dụng những document có độ tương đồng cao
                context_parts.append(text)
        return "\n\n".join(context_parts)

    def answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Tạo câu trả lời dựa trên câu hỏi và các document liên quan"""
        context = self._format_context(retrieved_docs)
        response = self.chain.run(context=context, question=question)
        return response