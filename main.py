from typing import Annotated
import os
import re
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from opensearchpy import OpenSearch

# --- Tên index OpenSearch ---
index_name = "chatbot_docs"

# --- Kết nối OpenSearch ---
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "Thaco@1234"),
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# --- Load model embedding (phải dùng cùng model như khi tạo embeddings) ---
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

# --- Load reranker ---
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
except:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def preprocess_text(text):
    """Tiền xử lý text để tách các từ dính liền nhau"""
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])([A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ])', r'\1 \2', text)
    text = re.sub(r'([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđA-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ])', r'\1 \2', text)
    return text

def search_similar(query_text, top_k=50):
    """Tìm kiếm kết hợp vector similarity và text search"""
    
    # Tiền xử lý query
    processed_query = preprocess_text(query_text)
    
    # Tạo embedding cho query
    query_embedding = model.encode([processed_query])[0].tolist()
    
    # Tìm kiếm kết hợp KNN và text search
    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": top_k
                            }
                        }
                    },
                    {
                        "match": {
                            "text": {
                                "query": processed_query,
                                "boost": 0.5
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "text": {
                                "query": processed_query,
                                "boost": 1.0
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["text", "chunk_index", "length"]
    }
    
    try:
        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        
        # Lấy text và tiền xử lý
        docs_with_score = []
        for hit in hits:
            text = preprocess_text(hit["_source"]["text"])
            score = hit["_score"]
            docs_with_score.append((text, score))
        
        # Sắp xếp theo score
        docs_with_score.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in docs_with_score]
        
        print(f"🔍 Tìm thấy {len(docs)} documents")
        return docs
        
    except Exception as e:
        print(f"❌ Lỗi search: {e}")
        # Fallback về KNN search đơn giản
        return search_knn_only(query_text, top_k)

def search_knn_only(query_text, top_k=50):
    """Fallback: chỉ sử dụng KNN search"""
    processed_query = preprocess_text(query_text)
    query_embedding = model.encode([processed_query])[0].tolist()
    
    search_query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        },
        "_source": ["text"]
    }
    
    try:
        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        docs = [preprocess_text(hit["_source"]["text"]) for hit in hits]
        print(f"🔍 KNN Fallback: Tìm thấy {len(docs)} documents")
        return docs
    except Exception as e:
        print(f"❌ Lỗi KNN search: {e}")
        return []

def keyword_relevance_score(query, doc):
    """Tính điểm relevance dựa trên từ khóa"""
    query_lower = query.lower()
    doc_lower = doc.lower()
    
    score = 0
    
    # Tìm tên người
    names = re.findall(r'\b[A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ][a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ]+(?:\s+[A-ZÀÁẢÃẠÂẦẤẨẪẬÊỀẾỂỄỆÔỒỐỔỖỘƯỪỨỬỮỰĐ][a-zàáảãạâầấẩẫậêềếểễệôồốổỗộưừứửữựđ]+)*', query)
    for name in names:
        if name.lower() in doc_lower:
            score += 3
    
    # Tìm năm sinh
    years = re.findall(r'\b(19|20)\d{2}\b', query)
    for year in years:
        if year in doc_lower:
            score += 2
    
    # Từ khóa liên quan đến sinh năm
    if any(word in query_lower for word in ["sinh", "năm", "born"]):
        if any(word in doc_lower for word in ["sinh", "năm", "born"]):
            score += 1
    
    # Các từ khóa quan trọng khác
    important_keywords = ["giám đốc", "chủ tịch", "tổng giám đốc", "phó", "trưởng", "ông", "bà"]
    for keyword in important_keywords:
        if keyword in query_lower and keyword in doc_lower:
            score += 1
    
    return score

def rerank_docs(query, docs, top_k=5):
    """Cải thiện reranking với kết hợp keyword matching và cross-encoder"""
    
    if not docs:
        return []
    
    # Bước 1: Lọc theo keyword relevance
    keyword_scores = [(doc, keyword_relevance_score(query, doc)) for doc in docs]
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top docs có keyword relevance cao
    high_keyword_docs = [doc for doc, score in keyword_scores[:15] if score > 0]
    remaining_docs = [doc for doc, score in keyword_scores if score == 0][:10]
    
    # Kết hợp
    docs_to_rerank = high_keyword_docs + remaining_docs
    
    # Bước 2: Sử dụng cross-encoder cho reranking
    if len(docs_to_rerank) == 0:
        docs_to_rerank = docs[:15]
    
    try:
        pairs = [[query, doc] for doc in docs_to_rerank]
        cross_scores = reranker.predict(pairs)
        
        # Kết hợp điểm keyword và cross-encoder
        final_scores = []
        for i, (doc, cross_score) in enumerate(zip(docs_to_rerank, cross_scores)):
            keyword_score = keyword_relevance_score(query, doc)
            # Trọng số: 70% cross-encoder, 30% keyword
            final_score = 0.7 * cross_score + 0.3 * keyword_score
            final_scores.append((doc, final_score, keyword_score, cross_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Debug info
        print(f"📊 Reranking scores:")
        for i, (doc, final_score, kw_score, cross_score) in enumerate(final_scores[:5], 1):
            print(f"  {i}. Final: {final_score:.3f} (KW: {kw_score}, Cross: {cross_score:.3f}) - {doc[:80]}...")
        
        return [doc for doc, _, _, _ in final_scores[:top_k]]
    
    except Exception as e:
        print(f"❌ Reranker error: {e}")
        # Fallback về keyword ranking
        return [doc for doc, score in keyword_scores[:top_k]]

# --- Định nghĩa State cho langgraph ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Cần set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"
llm = init_chat_model("google_genai:gemini-2.0-flash")

def chatbot(state: State):
    user_message = state["messages"][-1].content
    print(f"🤔 User query: {user_message}")

    # Tìm kiếm văn bản liên quan
    docs = search_similar(user_message, top_k=30)
    print(f"📚 Docs found: {len(docs)}")
    
    if not docs:
        return {"messages": [{"role": "assistant", "content": "Xin lỗi, tôi không tìm thấy thông tin liên quan trong dữ liệu."}]}
    
    # Rerank với thuật toán cải thiện
    reranked_docs = rerank_docs(user_message, docs, top_k=10)
    print(f"🏆 Top {len(reranked_docs)} reranked docs selected")
    
    # Lấy top 5 tài liệu để đưa vào prompt
    selected_docs = reranked_docs[:5]

    context = "\n---\n".join(selected_docs)
    prompt = f"""
    Dựa vào các tài liệu sau (mỗi dòng là một đoạn thông tin, có thể lặp, giữ nguyên thông tin thời gian, địa danh, tên tổ chức như trong tài liệu):

    {context}

    Hỏi: {user_message}

    Trả lời chính xác và súc tích, bao gồm đầy đủ thông tin thời gian (bao gồm năm nếu có trong tài liệu), loại bỏ các ký tự đặc biệt (như gạch dưới) trong tên người hoặc tổ chức để đảm bảo câu trả lời tự nhiên và đúng ngữ pháp tiếng Việt, không suy đoán từ việc dữ liệu có thể lặp lại. Nếu thông tin được hỏi không có trong tài liệu, trả lời: "Dữ liệu này mình chưa có thông tin."
    """

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return {"messages": [{"role": "assistant", "content": "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn."}]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def test_search():
    """Hàm test để kiểm tra search function"""
    print("🧪 Testing search function...")
    
    test_queries = [
        "Trần Bá Dương",
        "sinh năm 1960",
        "giám đốc điều hành"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Test query: '{query}'")
        docs = search_similar(query, top_k=5)
        if docs:
            print(f"✅ Found {len(docs)} docs")
            for i, doc in enumerate(docs[:2], 1):
                print(f"  {i}. {doc[:100]}...")
        else:
            print("❌ No docs found")

def chat_loop():
    print("🤖 Chat started! Type 'quit', 'exit', 'q', or 'test' to test search.")
    
    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == "test":
                test_search()
                continue
            
            if not user_input:
                continue
            
            print("🤖 Assistant: ", end="", flush=True)
            for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        print(value["messages"][-1].content)
                        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    # Kiểm tra kết nối OpenSearch
    try:
        info = client.info()
        print(f"✅ Connected to OpenSearch: {info['version']['number']}")
        
        # Kiểm tra index có tồn tại không
        if client.indices.exists(index=index_name):
            count = client.count(index=index_name)
            print(f"✅ Index '{index_name}' exists with {count['count']} documents")
        else:
            print(f"❌ Index '{index_name}' does not exist. Please run the embedding creation script first.")
            exit(1)
            
    except Exception as e:
        print(f"❌ Cannot connect to OpenSearch: {e}")
        exit(1)
    
    chat_loop()