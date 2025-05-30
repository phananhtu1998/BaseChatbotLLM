import sys
from .search_interface import SearchInterface

def main():
    """Hàm main để chạy chương trình."""
    print("🚀 Khởi động hệ thống tìm kiếm AI với Gemini...")
    print("⚠️  Lưu ý: Hệ thống sử dụng Google search thực tế")
    
    # Sử dụng API key được cung cấp  
    API_KEY = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"
    MODEL = "gemini-2.0-flash"
    
    try:
        search_interface = SearchInterface(API_KEY, MODEL)
        search_interface.start_interactive_search()
    except Exception as e:
        print(f"❌ Lỗi khởi động hệ thống: {e}")
        print("🔧 Vui lòng kiểm tra kết nối internet và API key")

def quick_demo():
    """Demo nhanh với một câu hỏi mẫu."""
    print("🎬 DEMO NHANH VỚI GEMINI AI")
    print("="*40)
    
    API_KEY = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"
    MODEL = "gemini-2.0-flash"
    
    try:
        interface = SearchInterface(API_KEY, MODEL)
        
        sample_questions = [
            "Python là gì và tại sao nên học?",
            "Cách học machine learning hiệu quả",
            "Xu hướng AI 2024"
        ]
        
        for question in sample_questions:
            print(f"\n📝 Demo câu hỏi: {question}")
            interface.perform_search(question, top_k=3)
            print("\n" + "="*60)
            input("Nhấn Enter để tiếp tục demo...")
            
    except Exception as e:
        print(f"❌ Lỗi demo: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        main() 