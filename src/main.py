from opensearchpy import OpenSearch
from src.config.config import OPENSEARCH_CONFIG, INDEX_NAME
from src.chat.chatbot import chat_loop

def main():
    # Initialize OpenSearch client
    client = OpenSearch(**OPENSEARCH_CONFIG)
    
    # Kiểm tra kết nối OpenSearch
    try:
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info['version']['number']}")
        
        # Kiểm tra index có tồn tại không
        if client.indices.exists(index=INDEX_NAME):
            count = client.count(index=INDEX_NAME)
            print(f"✅ Index '{INDEX_NAME}' exists with {count['count']} documents")
        else:
            print(f"❌ Index '{INDEX_NAME}' does not exist. Please run the embedding creation script first.")
            exit(1)
            
    except Exception as e:
        print(f"❌ Cannot connect to OpenSearch: {e}")
        exit(1)
    
    # Start chat loop
    chat_loop()

if __name__ == "__main__":
    main() 