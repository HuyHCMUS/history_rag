from typing import List
import re

class QueryTransformer:
    def __init__(self):
        self.question_patterns = [
            r"^(ai|who|what|when|where|why|how)",
            r"\?"
        ]

    def is_question(self, text: str) -> bool:
        text = text.lower().strip()
        return any(re.search(pattern, text) for pattern in self.question_patterns)

    def transform(self, query: str) -> str:
        """
        Transform questions into statements for better retrieval
        Example: "Who was the king of Ly Dynasty?" -> "king Ly Dynasty"
        """
        if not self.is_question(query):
            return query

        # Remove question words and punctuation
        query = query.lower()
        query = re.sub(r'^(ai|who|what|when|where|why|how)\s+', '', query)
        query = re.sub(r'\?', '', query)
        
        # Remove common words that don't add meaning
        stopwords = ['is', 'are', 'was', 'were', 'did', 'do', 'does']
        query = ' '.join(word for word in query.split() if word not in stopwords)

        return query.strip()