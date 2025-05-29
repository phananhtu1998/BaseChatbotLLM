from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType, utility, Index
)
from sentence_transformers import SentenceTransformer
from spacy.lang.vi import Vietnamese
import uuid

# 1. Kết nối tới Milvus
connections.connect("default", host="localhost", port="19530")

collection_name = "chatbot_docs"

# 2. Đọc dữ liệu và tách câu với spaCy
nlp = Vietnamese()
nlp.add_pipe("sentencizer")
with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
doc = nlp(raw_text)
sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# 3. Tạo embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

# 4. Tạo schema
text_id_field = FieldSchema(
    name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36
)
text_field = FieldSchema(
    name="text", dtype=DataType.VARCHAR, max_length=2048
)
vector_field = FieldSchema(
    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])
)
schema = CollectionSchema(
    fields=[text_id_field, text_field, vector_field],
    description="Text embeddings for chatbot"
)

# 5. Tạo collection nếu chưa có
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ Đã tạo collection '{collection_name}'")
else:
    collection = Collection(collection_name)
    print(f"ℹ️ Collection '{collection_name}' đã tồn tại")

# 6. Chuẩn bị dữ liệu chèn (lọc câu dài <= 512 ký tự)
filtered_sentences = []
filtered_embeddings = []
ids = []

for i, sent in enumerate(sentences):
    if len(sent) <= 512:
        filtered_sentences.append(sent)
        filtered_embeddings.append(embeddings[i].tolist())
        ids.append(str(uuid.uuid4()))
    else:
        print(f"⚠️ Bỏ qua câu quá dài ({len(sent)}): {sent[:60]}...")

data = [ids, filtered_sentences, filtered_embeddings]

# 7. Chèn dữ liệu
insert_result = collection.insert(data)
collection.flush()
print(f"✅ Đã chèn {len(filtered_sentences)} bản ghi vào collection")

# 8. Tạo index nếu chưa có
if not collection.has_index():
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",  # hoặc "L2" nếu bạn dùng L2 distance
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Đã tạo index cho collection")

print("🎉 Hoàn thành chuẩn bị dữ liệu và index.")
