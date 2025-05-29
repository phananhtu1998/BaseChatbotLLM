import os

# OpenSearch config
OPENSEARCH_CONFIG = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_auth": ("admin", "Thaco@1234"),
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
}

# Index name
INDEX_NAME = "chatbot_docs"

# Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"

# Model configs
EMBEDDING_MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2"
]

RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
]

# Search configs
SEARCH_TOP_K = 50
RERANK_TOP_K = 5
FINAL_TOP_K = 10

# Important keywords for relevance scoring
IMPORTANT_KEYWORDS = [
    "giám đốc", "chủ tịch", "tổng giám đốc", 
    "phó", "trưởng", "ông", "bà"
] 