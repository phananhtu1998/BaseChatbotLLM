import os
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime

class PostgreSQLChatbot:
    def __init__(self, db_config: Dict[str, str], gemini_api_key: str):
        """
        Khởi tạo chatbot với cấu hình database và API key
        
        Args:
            db_config: Dictionary chứa thông tin kết nối DB
            gemini_api_key: API key cho Gemini
        """
        self.db_config = db_config
        self.connection = None
        
        # Cấu hình Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Lưu trữ schema database
        self.db_schema = {}
        
        # Kết nối database
        self.connect_db()
        
        # Lấy schema database
        self.load_db_schema()
    
    def connect_db(self):
        """Kết nối đến PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config.get('port', 5432)
            )
            print("✅ Kết nối database thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối database: {e}")
            raise
    
    def load_db_schema(self):
        """Tải schema của database"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Lấy danh sách bảng
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = cursor.fetchall()
                
                # Lấy thông tin cột cho mỗi bảng
                for table in tables:
                    table_name = table['table_name']
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position
                    """, (table_name,))
                    
                    columns = cursor.fetchall()
                    self.db_schema[table_name] = columns
                
                print(f"✅ Đã tải schema cho {len(self.db_schema)} bảng")
                
        except Exception as e:
            print(f"❌ Lỗi tải schema: {e}")
    
    def get_schema_context(self) -> str:
        """Tạo context về schema database cho Gemini"""
        schema_text = "DATABASE SCHEMA:\n"
        
        for table_name, columns in self.db_schema.items():
            schema_text += f"\nBảng: {table_name}\n"
            schema_text += "Cột:\n"
            
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                schema_text += f"  - {col['column_name']}: {col['data_type']} {nullable}{default}\n"
        
        return schema_text
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Thực thi SQL query và trả về kết quả"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                
                # Kiểm tra nếu là SELECT query
                if query.strip().upper().startswith('SELECT'):
                    results = cursor.fetchall()
                    return {
                        'success': True,
                        'data': [dict(row) for row in results],
                        'row_count': len(results)
                    }
                else:
                    # Cho INSERT, UPDATE, DELETE
                    self.connection.commit()
                    return {
                        'success': True,
                        'message': f'Query thực hiện thành công. Số dòng bị ảnh hưởng: {cursor.rowcount}',
                        'row_count': cursor.rowcount
                    }
                    
        except Exception as e:
            self.connection.rollback()
            return {
                'success': False,
                'error': str(e)
            }
    
    def format_results(self, results: List[Dict], limit: int = 20) -> str:
        """Format kết quả query thành dạng dễ đọc cho hệ thống bán hàng"""
        if not results:
            return "Không có dữ liệu."
        
        # Giới hạn số dòng hiển thị
        display_results = results[:limit]
        
        formatted = f"Tìm thấy {len(results)} kết quả"
        if len(results) > limit:
            formatted += f" (hiển thị {limit} đầu tiên)"
        formatted += ":\n\n"
        
        for i, row in enumerate(display_results, 1):
            formatted += f"{i}. "
            row_text = ""
            
            for key, value in row.items():
                # Format đặc biệt cho tiền tệ
                if key in ['price', 'total_amount', 'unit_price', 'revenue', 'doanh_thu', 'tong_tien']:
                    if value is not None:
                        formatted_price = f"{int(value):,}".replace(',', '.')
                        row_text += f"{key}: {formatted_price} VNĐ, "
                    else:
                        row_text += f"{key}: 0 VNĐ, "
                
                # Format đặc biệt cho ngày tháng
                elif key in ['order_date', 'created_at', 'updated_at']:
                    if value is not None:
                        if hasattr(value, 'strftime'):
                            row_text += f"{key}: {value.strftime('%d/%m/%Y %H:%M')}, "
                        else:
                            row_text += f"{key}: {value}, "
                    else:
                        row_text += f"{key}: N/A, "
                
                # Format cho số lượng
                elif key in ['quantity', 'stock', 'so_luong', 'ton_kho']:
                    row_text += f"{key}: {value} cái, "
                
                # Các trường khác
                else:
                    row_text += f"{key}: {value}, "
            
            formatted += row_text.rstrip(", ") + "\n"
        
        return formatted
    
    async def generate_sql_query(self, user_question: str) -> str:
        """Sử dụng Gemini để tạo SQL query từ câu hỏi tự nhiên"""
        
        schema_context = self.get_schema_context()
        
        prompt = f"""
Bạn là một chuyên gia SQL cho hệ thống bán hàng. Nhiệm vụ của bạn là chuyển đổi câu hỏi tiếng Việt thành SQL query PostgreSQL.

{schema_context}

BUSINESS CONTEXT:
- Đây là hệ thống bán hàng với các bảng: categories (danh mục), products (sản phẩm), customers (khách hàng), orders (đơn hàng), order_items (chi tiết đơn hàng)
- Giá tiền được lưu dưới dạng NUMERIC không có phần thập phân (VD: 29990000 = 29,990,000 VNĐ)
- Status đơn hàng: 'pending', 'shipped', 'cancelled'
- Khi tính tổng tiền, sử dụng quantity * unit_price từ order_items

COMMON QUERIES PATTERNS:
- Doanh thu: SUM(oi.quantity * oi.unit_price) FROM order_items oi JOIN orders o ON oi.order_id = o.id
- Top sản phẩm bán chạy: SUM(quantity) FROM order_items GROUP BY product_id
- Khách hàng VIP: theo tổng tiền đã mua
- Báo cáo theo thời gian: sử dụng DATE_TRUNC hoặc EXTRACT từ order_date

RULES:
1. Chỉ trả về SQL query, không giải thích
2. Sử dụng PostgreSQL syntax
3. Đảm bảo query an toàn, không có SQL injection
4. Sử dụng JOIN phù hợp để lấy tên thay vì ID
5. Với tiền tệ, format theo định dạng Việt Nam
6. Sử dụng LIMIT 20 mặc định trừ khi hỏi cụ thể
7. Với thống kê thời gian, ưu tiên đơn hàng đã shipped
8. Khi so sánh chuỗi (như category, tên sản phẩm, trạng thái...), hãy dùng ILIKE hoặc LOWER(...) để không phân biệt chữ hoa/thường


Câu hỏi: {user_question}

SQL Query:
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            sql_query = response.text.strip()
            
            # Loại bỏ markdown formatting nếu có
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            
            return sql_query.strip()
            
        except Exception as e:
            raise Exception(f"Lỗi tạo SQL query: {e}")
    
    async def generate_response(self, user_question: str, query_result: Dict[str, Any]) -> str:
        """Tạo phản hồi tự nhiên từ kết quả query"""
        
        if not query_result['success']:
            return f"Xin lỗi, có lỗi xảy ra khi thực hiện truy vấn: {query_result['error']}"
        
        # Nếu là SELECT query
        if 'data' in query_result:
            data_summary = self.format_results(query_result['data'])
            
            prompt = f"""
Dựa trên câu hỏi của user và kết quả truy vấn, hãy tạo một phản hồi tự nhiên bằng tiếng Việt.

Câu hỏi: {user_question}

Kết quả truy vấn:
{data_summary}

Hãy trả lời một cách tự nhiên, dễ hiểu và hữu ích. Nếu có nhiều kết quả, hãy tóm tắt hoặc highlight những điểm quan trọng.
"""
        else:
            prompt = f"""
Câu hỏi: {user_question}
Kết quả: {query_result['message']}

Hãy tạo phản hồi tự nhiên bằng tiếng Việt cho thao tác này.
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Đã thực hiện thành công nhưng có lỗi tạo phản hồi: {e}"
    
    async def chat(self, user_question: str) -> str:
        """Xử lý câu hỏi của user và trả về phản hồi"""
        try:
            print(f"🤔 Đang xử lý câu hỏi: {user_question}")
            
            # Tạo SQL query
            sql_query = await self.generate_sql_query(user_question)
            print(f"🔍 SQL Query: {sql_query}")
            
            # Thực thi query
            result = self.execute_query(sql_query)
            
            # Tạo phản hồi
            response = await self.generate_response(user_question, result)
            
            return response
            
        except Exception as e:
            return f"Xin lỗi, có lỗi xảy ra: {e}"
    
    def close(self):
        """Đóng kết nối database"""
        if self.connection:
            self.connection.close()
            print("✅ Đã đóng kết nối database")

# Demo sử dụng
async def main():
    # Cấu hình database - CẬP NHẬT THEO THÔNG TIN CỦA BẠN
    db_config = {
        'host': 'localhost',
        'database': 'db_test',  # Tên database của bạn
        'user': 'admin',      # Username của bạn
        'password': '123',  # Password của bạn
        'port': 5432
    }
    gemini_api_key = "AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0"
    
    try:
        # Khởi tạo chatbot
        chatbot = PostgreSQLChatbot(db_config, gemini_api_key)
        
        print("🤖 Chatbot Hệ thống Bán hàng đã sẵn sàng!")
        print("📊 Dữ liệu hiện có:")
        print("   - Danh mục: Điện thoại, Thời trang, Đồ gia dụng")
        print("   - Sản phẩm: iPhone 15 Pro, Áo thun nam, Máy hút bụi")
        print("   - Khách hàng: Nguyễn Văn A, Trần Thị B")
        print("   - Đơn hàng: 2 đơn hàng mẫu")
        print("\n💡 Ví dụ câu hỏi:")
        print("   - 'Doanh thu tháng 6 là bao nhiêu?'")
        print("   - 'Sản phẩm nào bán chạy nhất?'")
        print("   - 'Khách hàng nào mua nhiều nhất?'")
        print("   - 'Còn bao nhiêu iPhone trong kho?'")
        print("   - 'Liệt kê tất cả đơn hàng chưa giao'")
        print("\nNhập 'quit' để thoát\n")
        
        while True:
            user_input = input("Bạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'thoát']:
                break
            
            if not user_input:
                continue
            
            print("🤖 Đang suy nghĩ...")
            response = await chatbot.chat(user_input)
            print(f"Bot: {response}\n")
    
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        if 'chatbot' in locals():
            chatbot.close()

# Chạy demo
if __name__ == "__main__":
    asyncio.run(main())