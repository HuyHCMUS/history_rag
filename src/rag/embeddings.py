# Embedding generation
from sentence_transformers import SentenceTransformer, models
import numpy as np

def create_embeddings(chunks, model_name="vinai/phobert-base-v2"):
    """
    Tạo embeddings cho các đoạn văn bản sử dụng PhoBERT với mean pooling.
    
    Args:
        chunks (list): Danh sách các đoạn văn bản.
        model_name (str): Tên của mô hình PhoBERT.
    
    Returns:
        numpy.ndarray: Ma trận embeddings.
    """
    # Load PhoBERT với transformer model
    word_embedding_model = models.Transformer(model_name, max_seq_length=256)

    # Thêm lớp mean pooling
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

    # Kết hợp thành một SentenceTransformer model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Tạo embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)

    return np.array(embeddings)  # Chuyển về numpy array
