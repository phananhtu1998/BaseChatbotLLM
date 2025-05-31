from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch
import os

# ✅ Thiết lập API Key cho Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"  # thay bằng key của bạn
search_with_tavily = TavilySearch(max_results=3, tavily_api_key="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")

# ✅ Định nghĩa tool bằng tiếng Việt
@tool
def search_documents(query: str) -> str:
    """
    Dùng công cụ này để tìm kiếm trong các tài liệu nội bộ, như chính sách công ty, báo cáo nội bộ,
    hoặc thông tin về nhân sự. Thích hợp khi người dùng hỏi về thông tin nội bộ không có trên internet.
    """
    return f"[Tìm trong TÀI LIỆU NỘI BỘ] Đang tìm với truy vấn: {query}\n✅ Kết quả mô phỏng: Đã tìm thấy thông tin liên quan đến '{query}' trong hệ thống tài liệu nội bộ."

@tool
def search_web(query: str) -> str:
    """
    Dùng công cụ này để tìm kiếm thông tin công khai từ Internet, như tin tức, sự kiện, người nổi tiếng
    hoặc kiến thức phổ thông. Thích hợp cho các câu hỏi về thông tin công cộng.
    """
    try:
        # Gọi Tavily search
        search_results = search_with_tavily.invoke(query)
        
        # Xử lý kết quả từ Tavily
        if isinstance(search_results, list) and len(search_results) > 0:
            # Nếu trả về list của các kết quả
            formatted_results = []
            for i, result in enumerate(search_results[:3], 1):  # Lấy tối đa 3 kết quả
                if isinstance(result, dict):
                    title = result.get('title', 'Không có tiêu đề')
                    content = result.get('content', result.get('snippet', 'Không có nội dung'))
                    url = result.get('url', '')
                    formatted_results.append(f"Kết quả {i}:\n- Tiêu đề: {title}\n- Nội dung: {content}\n- URL: {url}")
                else:
                    formatted_results.append(f"Kết quả {i}: {str(result)}")
            
            return f"[Tìm trên WEB] Truy vấn: {query}\n\n" + "\n\n".join(formatted_results)
        
        elif isinstance(search_results, dict):
            # Nếu trả về dict đơn lẻ
            title = search_results.get('title', 'Không có tiêu đề')
            content = search_results.get('content', search_results.get('snippet', 'Không có nội dung'))
            url = search_results.get('url', '')
            return f"[Tìm trên WEB] Truy vấn: {query}\n\nKết quả:\n- Tiêu đề: {title}\n- Nội dung: {content}\n- URL: {url}"
        
        elif isinstance(search_results, str):
            # Nếu trả về string
            return f"[Tìm trên WEB] Truy vấn: {query}\n\nKết quả: {search_results}"
        
        else:
            # Trường hợp khác, convert sang string
            return f"[Tìm trên WEB] Truy vấn: {query}\n\nKết quả: {str(search_results)}"
            
    except Exception as e:
        return f"[LỖI TÌM KIẾM WEB] Không thể tìm kiếm '{query}'. Lỗi: {str(e)}"

# ✅ Khởi tạo mô hình Gemini và gắn tools
llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0)
tools = [search_documents, search_web]
llm_with_tools = llm.bind_tools(tools)

# ✅ Tạo dictionary để ánh xạ tên tool với function
tool_map = {
    "search_documents": search_documents,
    "search_web": search_web
}

# ✅ System message cải thiện với hướng dẫn đa ngôn ngữ
system_message = SystemMessage(
    content="""
Bạn là một trợ lý AI thông minh và đa ngôn ngữ. Bạn có 2 công cụ:

- `search_web`: dùng để tìm tin tức công khai như giá vàng, người nổi tiếng, sự kiện, kiến thức phổ thông từ Internet.
- `search_documents`: dùng để tìm thông tin nội bộ như nội quy công ty, chính sách công ty, báo cáo nội bộ.

QUAN TRỌNG - Quy tắc trả lời:
1. LUÔN trả lời bằng chính xác ngôn ngữ mà người dùng sử dụng trong câu hỏi
2. Nếu câu hỏi bằng tiếng Việt → trả lời bằng tiếng Việt
3. Nếu câu hỏi bằng tiếng Anh → trả lời bằng tiếng Anh
4. Nếu câu hỏi bằng ngôn ngữ khác → trả lời bằng ngôn ngữ đó

Chọn công cụ phù hợp:
- Câu hỏi về thông tin công khai (tin tức, người nổi tiếng, sự kiện) → `search_web`
- Câu hỏi về thông tin nội bộ công ty → `search_documents`

Sau khi nhận được kết quả từ công cụ, hãy:
- Tổng hợp thông tin một cách tự nhiên
- Trả lời đầy đủ và hữu ích
- Giữ nguyên ngôn ngữ của câu hỏi
"""
)

# ✅ Vòng lặp nhập từ người dùng
print("💬 Chatbot đa ngôn ngữ sẵn sàng. Gõ câu hỏi bằng bất kỳ ngôn ngữ nào và nhấn Enter. Nhấn Ctrl+C để thoát.\n")

try:
    while True:
        query = input("🧑 Bạn: ")
        if not query.strip():
            continue
            
        # Tạo conversation với system message và user message
        messages = [system_message, HumanMessage(content=query)]
        
        print("📨 Đang xử lý...")
        
        # Gọi AI với tools
        response = llm_with_tools.invoke(messages)
        
        # Kiểm tra xem AI có gọi tool không
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🛠️  Sử dụng công cụ: {response.tool_calls[0]['name']}")
            
            # Thực thi từng tool call
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']
                
                print(f"⚙️  Đang tìm kiếm: {tool_args.get('query', 'N/A')}")
                
                # Gọi tool function
                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                        
                        # Tạo ToolMessage với kết quả
                        tool_messages.append(
                            ToolMessage(
                                content=tool_result,
                                tool_call_id=tool_id
                            )
                        )
                    except Exception as e:
                        print(f"❌ Lỗi khi chạy tool {tool_name}: {e}")
                        tool_messages.append(
                            ToolMessage(
                                content=f"Lỗi tìm kiếm: {str(e)}",
                                tool_call_id=tool_id
                            )
                        )
            
            # Thêm tool response vào conversation và gọi AI lần nữa để có câu trả lời cuối cùng
            if tool_messages:
                messages.extend([response] + tool_messages)
                final_response = llm_with_tools.invoke(messages)
                print(f"🤖 Trợ lý: {final_response.content}\n")
            else:
                print("🤖 Trợ lý: Không thể thực thi công cụ tìm kiếm.\n")
        else:
            # AI trả lời trực tiếp mà không cần tool
            print(f"🤖 Trợ lý: {response.content}\n")
            
except KeyboardInterrupt:
    print("\n👋 Tạm biệt!")
except Exception as e:
    print(f"\n❌ Lỗi hệ thống: {e}")
    print("👋 Tạm biệt!")