from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from spacy.lang.vi import Vietnamese
import uuid
import json
import re
from typing import List, Dict
import hashlib

# Kết nối OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    http_auth=("admin", "Thaco@1234"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

index_name = "chatbot_docs"

# Cấu hình splitting
MIN_CHUNK_SIZE = 50      
MAX_CHUNK_SIZE = 300     
OVERLAP_SIZE = 50        
MIN_SENTENCE_LENGTH = 10 

def preprocess_text(text: str) -> str:
    """Tiền xử lý text để cải thiện chất lượng"""
    # Loại bỏ ký tự không cần thiết
    text = re.sub(r'\s+', ' ', text)  # Chuẩn hóa khoảng trắng
    
    # Tách các từ dính liền (như sinhnăm1960 -> sinh năm 1960)
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])([A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ])', r'\1 \2', text)
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])', r'\1 \2', text)
    
    return text.strip()

def smart_text_split(text: str) -> List[str]:
    """Chia text thành chunks thông minh với overlap"""
    
    # Tiền xử lý text
    processed_text = preprocess_text(text)
    
    # Khởi tạo spaCy
    nlp = Vietnamese()
    nlp.add_pipe("sentencizer")
    
    # Tách câu
    doc = nlp(processed_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= MIN_SENTENCE_LENGTH]
    
    print(f"📊 Đã tách được {len(sentences)} câu")
    
    # Tạo chunks với overlap
    chunks = []
    current_chunk = ""
    current_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Kiểm tra xem có nên tạo chunk mới không
        if (len(current_chunk) + len(sentence) > MAX_CHUNK_SIZE and 
            len(current_chunk) >= MIN_CHUNK_SIZE):
            
            # Lưu chunk hiện tại
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Tạo overlap: giữ lại 1-2 câu cuối
            overlap_sentences = current_sentences[-1:] if current_sentences else []
            overlap_text = " ".join(overlap_sentences)
            
            if overlap_text:
                current_chunk = overlap_text + " " + sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk = sentence
                current_sentences = [sentence]
        else:
            # Thêm câu vào chunk hiện tại
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_sentences.append(sentence)
    
    # Thêm chunk cuối cùng
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    # Lọc chunks quá ngắn
    chunks = [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_SIZE]
    
    print(f"📦 Tạo được {len(chunks)} chunks")
    print(f"📏 Độ dài chunk: min={min(len(c) for c in chunks)}, max={max(len(c) for c in chunks)}, avg={sum(len(c) for c in chunks)/len(chunks):.1f}")
    
    return chunks

def create_embeddings():
    """Tạo embeddings đơn giản và tương thích"""
    print("🚀 Bắt đầu tạo embeddings...")
    
    # Load model
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Sử dụng model paraphrase-multilingual-MiniLM-L12-v2")
    except:
        try:
            model = SentenceTransformer("all-MiniLM-L12-v2") 
            print("✅ Sử dụng model all-MiniLM-L12-v2")
        except:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Sử dụng model all-MiniLM-L6-v2")
    
    # Đọc dữ liệu
    print("📖 Đọc file...")
    with open("data.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print(f"📄 File gốc: {len(raw_text):,} ký tự")
    
    # Chia text thành chunks
    chunks = smart_text_split(raw_text)
    
    if not chunks:
        print("❌ Không có chunks nào được tạo!")
        return
    
    # Tạo embeddings
    print("🔢 Tạo embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"✅ Tạo được {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
    
    # Tạo index OpenSearch đơn giản
    create_simple_index(len(embeddings[0]))
    
    # Chuẩn bị documents
    print("📦 Chuẩn bị documents...")
    documents = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = f"chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
        
        document = {
            "_index": index_name,
            "_id": doc_id,
            "_source": {
                "id": doc_id,
                "text": chunk,
                "embedding": embedding.tolist(),
                "chunk_index": i,
                "length": len(chunk)
            }
        }
        documents.append(document)
    
    # Insert vào OpenSearch
    print("💾 Chèn dữ liệu...")
    from opensearchpy.helpers import bulk
    
    try:
        success, failed = bulk(client, documents, chunk_size=50)
        print(f"✅ Thành công: {success} documents")
        if failed:
            print(f"❌ Thất bại: {len(failed)} documents")
    except Exception as e:
        print(f"❌ Lỗi insert: {e}")
        return
    
    print("🎉 Hoàn thành!")
    print(f"📊 Thống kê cuối:")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Embedding dimension: {len(embeddings[0])}")
    print(f"   - Độ dài trung bình: {sum(len(c) for c in chunks) / len(chunks):.1f} ký tự")

def create_simple_index(embedding_dim):
    """Tạo index OpenSearch đơn giản"""
    
    # Xóa index cũ nếu tồn tại
    if client.indices.exists(index=index_name):
        print(f"🗑️ Xóa index cũ...")
        client.indices.delete(index=index_name)
    
    # Tạo index mới với cấu hình tối thiểu
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim
                },
                "chunk_index": {"type": "integer"},
                "length": {"type": "integer"}
            }
        }
    }
    
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"✅ Tạo index '{index_name}' thành công")
    except Exception as e:
        print(f"❌ Lỗi tạo index: {e}")
        raise

if __name__ == "__main__":
    create_embeddings()