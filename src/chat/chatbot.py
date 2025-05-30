from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from src.search.search_engine import search_similar, rerank_docs
from src.config.config import FINAL_TOP_K
from src.tools.search_tools import SmartSearchTool

# --- Định nghĩa State cho langgraph ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize LLM
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Initialize SmartSearchTool
smart_search = SmartSearchTool(api_key="your_gemini_api_key")

# Bind tools to LLM
llm_with_tools = llm.bind_tools([smart_search])

def chatbot(state: State):
    """Node xử lý chat và quyết định có sử dụng tool hay không"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def create_chat_graph():
    """Create and compile the chat graph with tools"""
    # Create graph
    graph_builder = StateGraph(State)
    
    # Add chatbot node
    graph_builder.add_node("chatbot", chatbot)
    
    # Add tools node
    tool_node = ToolNode(tools=[smart_search])
    graph_builder.add_node("tools", tool_node)
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # Add edges
    graph_builder.add_edge("tools", "chatbot")  # Return to chatbot after tool use
    graph_builder.add_edge(START, "chatbot")    # Start with chatbot
    
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