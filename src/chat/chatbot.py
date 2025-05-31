from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from src.search.search_engine import search_similar, rerank_docs
from src.config.config import FINAL_TOP_K

# --- Định nghĩa State cho langgraph ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
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
    
    # Lấy top 10 tài liệu để đưa vào prompt
    selected_docs = reranked_docs[:10]

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

def create_chat_graph():
    """Create and compile the chat graph"""
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    return graph_builder.compile()

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
    """Main chat loop"""
    print("🤖 Chat started! Type 'quit', 'exit', 'q', or 'test' to test search.")
    
    # Create chat graph
    graph = create_chat_graph()
    
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
    chat_loop()