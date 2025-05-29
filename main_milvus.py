from typing import Annotated
import os
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import CrossEncoder

# Load reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_docs(query, docs):
    # docs là list các đoạn text lấy từ Milvus
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)  # trả về score cho từng cặp
    # ghép điểm với docs
    scored_docs = list(zip(docs, scores))
    # sắp xếp giảm dần theo điểm
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    # trả về docs đã sắp xếp
    return [doc for doc, score in scored_docs]

# --- Kết nối Milvus và tải collection ---
connections.connect("default", host="localhost", port="19530")
collection_name = "chatbot_docs"
collection = Collection(collection_name)

# Kiểm tra index, nếu chưa có thì tạo (đảm bảo ko lỗi khi chạy)
if not collection.has_index():
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ Đã tạo index cho collection")

collection.load()  # Bắt buộc load trước khi search

# --- Tải model embedding ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Hàm chuẩn hóa vector (nếu dùng IP, normalize để giống cosine) ---
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

# --- Hàm tìm kiếm tương tự ---
def search_similar(query_text, top_k=3):
    query_embedding = model.encode([query_text])[0]
    query_embedding_norm = normalize(query_embedding).tolist()

    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }

    results = collection.search(
        data=[query_embedding_norm],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"],
    )
    hits = results[0]
    return [hit.entity.get("text") for hit in hits]

# --- Định nghĩa State cho langgraph ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

os.environ["GOOGLE_API_KEY"] = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"
llm = init_chat_model("google_genai:gemini-2.0-flash")

# --- Hàm chatbot với bước search trước khi gọi LLM ---
def chatbot(state: State):
    user_message = state["messages"][-1].content  # Lấy câu hỏi cuối cùng của user
    
    # Tìm kiếm văn bản liên quan
    docs = search_similar(user_message, top_k=50)
    #print("Docs:", docs)
    # Rerank lại top 10
    reranked_docs = rerank_docs(user_message, docs)
    
    # Lấy 3 tài liệu top đầu sau rerank để đưa vào prompt
    selected_docs = reranked_docs[:3]
    #print("selected_docs:", selected_docs)
    # Ghép tài liệu vào prompt cho LLM
    context = "\n---\n".join(selected_docs)
    prompt = f"""
    Dựa vào các tài liệu sau (mỗi dòng là một đoạn thông tin, có thể lặp, giữ nguyên thông tin thời gian, địa danh, tên tổ chức như trong tài liệu):
    {context}
    Hỏi: {user_message}
    Trả lời chính xác và súc tích, bao gồm đầy đủ thông tin thời gian (bao gồm năm nếu có trong tài liệu), loại bỏ các ký tự đặc biệt (như gạch dưới) trong tên người hoặc tổ chức để đảm bảo câu trả lời tự nhiên và đúng ngữ pháp tiếng Việt, không suy đoán từ việc dữ liệu có thể lặp lại. Nếu thông tin được hỏi không có trong tài liệu, trả lời: "Dữ liệu này mình chưa có thông tin.
    """

    # Gọi LLM
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def chat_loop():
    print("Chat started! Type 'quit', 'exit', or 'q' to end the conversation.")
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        print("Assistant:", value["messages"][-1].content)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

if __name__ == "__main__":
    chat_loop()
