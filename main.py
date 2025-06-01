from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_tavily import TavilySearch
import os
import base64
from PIL import Image
import io

# ✅ Thiết lập API Key cho Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"  # thay bằng key của bạn
search_with_tavily = TavilySearch(max_results=3, tavily_api_key="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")

# ✅ Tool phân tích hình ảnh
@tool
def analyze_image(image_path: str, question: str = "") -> str:
    """
    Phân tích hình ảnh và trả lời câu hỏi về hình ảnh. 
    Có thể mô tả nội dung, nhận diện đối tượng, đọc text trong ảnh, hoặc trả lời câu hỏi cụ thể về hình ảnh.
    
    Args:
        image_path: Đường dẫn đến file hình ảnh
        question: Câu hỏi cụ thể về hình ảnh (tùy chọn)
    """
    try:
        # Kiểm tra file có tồn tại
        if not os.path.exists(image_path):
            return f"❌ Không tìm thấy file hình ảnh: {image_path}"
        
        # Đọc và encode hình ảnh
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_b64 = base64.b64encode(image_data).decode()
        
        # Tạo model với khả năng xử lý hình ảnh
        vision_llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0)
        
        # Tạo prompt cho phân tích hình ảnh
        if question:
            prompt = f"Hãy phân tích hình ảnh này và trả lời câu hỏi: {question}"
        else:
            prompt = "Hãy mô tả chi tiết nội dung của hình ảnh này, bao gồm các đối tượng, màu sắc, hoạt động, và bất kỳ text nào có trong ảnh."
        
        # Tạo message với hình ảnh
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        )
        
        # Gọi AI để phân tích
        response = vision_llm.invoke([message])
        
        return f"[PHÂN TÍCH HÌNH ẢNH] {image_path}\n\n{response.content}"
        
    except Exception as e:
        return f"❌ Lỗi khi phân tích hình ảnh: {str(e)}"

# ✅ Tool tìm kiếm tài liệu nội bộ
@tool
def search_documents(query: str) -> str:
    """
    Dùng công cụ này để tìm kiếm trong các tài liệu nội bộ, như chính sách công ty, báo cáo nội bộ,
    hoặc thông tin về nhân sự. Thích hợp khi người dùng hỏi về thông tin nội bộ không có trên internet.
    """
    return f"[Tìm trong TÀI LIỆU NỘI BỘ] Đang tìm với truy vấn: {query}\n✅ Kết quả mô phỏng: Đã tìm thấy thông tin liên quan đến '{query}' trong hệ thống tài liệu nội bộ."

# ✅ Tool tìm kiếm web
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
tools = [search_documents, search_web, analyze_image]
llm_with_tools = llm.bind_tools(tools)

# ✅ Tạo dictionary để ánh xạ tên tool với function
tool_map = {
    "search_documents": search_documents,
    "search_web": search_web,
    "analyze_image": analyze_image
}

# ✅ System message cải thiện với hướng dẫn về hình ảnh
system_message = SystemMessage(
    content="""
Bạn là một trợ lý AI thông minh và đa ngôn ngữ. Bạn có 3 công cụ:

- search_web: dùng để tìm tin tức công khai như giá vàng, người nổi tiếng, sự kiện, kiến thức phổ thông từ Internet.
- search_documents: dùng để tìm thông tin nội bộ như nội quy công ty, chính sách công ty, báo cáo nội bộ.
- analyze_image: dùng để phân tích hình ảnh, mô tả nội dung, nhận diện đối tượng, đọc text trong ảnh, hoặc trả lời câu hỏi về hình ảnh.

QUAN TRỌNG - Quy tắc trả lời:
1. LUÔN trả lời bằng chính xác ngôn ngữ mà người dùng sử dụng trong câu hỏi
2. Nếu câu hỏi bằng tiếng Việt → trả lời bằng tiếng Việt
3. Nếu câu hỏi bằng tiếng Anh → trả lời bằng tiếng Anh
4. Nếu câu hỏi bằng ngôn ngữ khác → trả lời bằng ngôn ngữ đó

Chọn công cụ phù hợp:
- Câu hỏi về thông tin công khai (tin tức, người nổi tiếng, sự kiện) → search_web
- Câu hỏi về thông tin nội bộ công ty → search_documents
- Câu hỏi về hình ảnh (mô tả, phân tích, đọc text) → analyze_image

Đối với hình ảnh:
- Khi người dùng đề cập đến file hình ảnh, hãy sử dụng analyze_image
- Truyền đường dẫn file và câu hỏi cụ thể (nếu có)
- Nếu không có câu hỏi cụ thể, hãy mô tả tổng quan hình ảnh

Sau khi nhận được kết quả từ công cụ, hãy:
- Tổng hợp thông tin một cách tự nhiên
- Trả lời đầy đủ và hữu ích
- Giữ nguyên ngôn ngữ của câu hỏi
"""
)

# ✅ Hàm xử lý upload hình ảnh
def handle_image_upload():
    """Hướng dẫn người dùng upload hình ảnh"""
    print("\n📸 Để phân tích hình ảnh, bạn có thể:")
    print("1. Đặt file hình ảnh trong cùng thư mục với script này")
    print("2. Nhập tên file (ví dụ: 'image.jpg', 'photo.png')")
    print("3. Hoặc nhập đường dẫn đầy đủ (ví dụ: 'C:/Users/Desktop/image.jpg')")
    print("4. Sau đó hỏi về hình ảnh: 'Phân tích file image.jpg' hoặc 'Hình ảnh này có gì?'")
    print("\n💡 Ví dụ câu hỏi:")
    print("- 'Mô tả hình ảnh trong file photo.jpg'")
    print("- 'Đọc text in trong ảnh document.png'") 
    print("- 'Phân tích màu sắc trong image.jpg'")
    print("- 'Có bao nhiêu người trong ảnh family.jpg?'\n")

# ✅ Vòng lặp nhập từ người dùng
print("💬 Chatbot đa ngôn ngữ với phân tích hình ảnh sẵn sàng!")
print("🔍 Có thể tìm kiếm web, tài liệu nội bộ, và phân tích hình ảnh")
print("📝 Gõ câu hỏi bằng bất kỳ ngôn ngữ nào và nhấn Enter")
print("📸 Gõ 'help image' để xem hướng dẫn upload hình ảnh")
print("🚪 Nhấn Ctrl+C để thoát\n")

try:
    while True:
        query = input("🧑 Bạn: ")
        if not query.strip():
            continue
            
        # Kiểm tra lệnh help
        if query.lower() in ['help image', 'help img', 'hướng dẫn ảnh']:
            handle_image_upload()
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
                
                if tool_name == 'analyze_image':
                    print(f"📸 Đang phân tích hình ảnh: {tool_args.get('image_path', 'N/A')}")
                else:
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
                                content=f"Lỗi thực thi: {str(e)}",
                                tool_call_id=tool_id
                            )
                        )
            
            # Thêm tool response vào conversation và gọi AI lần nữa để có câu trả lời cuối cùng
            if tool_messages:
                messages.extend([response] + tool_messages)
                final_response = llm_with_tools.invoke(messages)
                print(f"🤖 Trợ lý: {final_response.content}\n")
            else:
                print("🤖 Trợ lý: Không thể thực thi công cụ.\n")
        else:
            # AI trả lời trực tiếp mà không cần tool
            print(f"🤖 Trợ lý: {response.content}\n")
            
except KeyboardInterrupt:
    print("\n👋 Tạm biệt!")
except Exception as e:
    print(f"\n❌ Lỗi hệ thống: {e}")
    print("👋 Tạm biệt!")