from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config.config import EMBEDDING_MODELS, RERANKER_MODELS

def load_embedding_model():
    """Load embedding model with fallback options"""
    for model_name in EMBEDDING_MODELS:
        try:
            model = SentenceTransformer(model_name)
            print(f"✅ Sử dụng model {model_name}")
            return model
        except:
            continue
    raise Exception("Không thể load bất kỳ model embedding nào")

def load_reranker():
    """Load reranker model with fallback options"""
    for model_name in RERANKER_MODELS:
        try:
            reranker = CrossEncoder(model_name)
            return reranker
        except:
            continue
    raise Exception("Không thể load bất kỳ model reranker nào")

# Initialize models
model = load_embedding_model()
reranker = load_reranker() 