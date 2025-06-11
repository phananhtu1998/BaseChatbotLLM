import psycopg2
from psycopg2.extras import RealDictCursor

def setup_database():
    """Khởi tạo database với dữ liệu mẫu"""
    
    # Cấu hình kết nối - CẬP NHẬT THEO THÔNG TIN CỦA BẠN
    db_config = {
        'host': 'localhost',
        'database': 'db_test',  # Tên database
        'user': 'admin',      # Username
        'password': '123',  # Password
        'port': 5432
    }
    
    # SQL script tạo database
    sql_script = """
-- XÓA BẢNG NẾU TỒN TẠI
DROP TABLE IF EXISTS order_items, orders, customers, products, categories CASCADE;

-- BẢNG DANH MỤC SẢN PHẨM
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

-- BẢNG SẢN PHẨM
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(12, 0) NOT NULL,
    stock INTEGER NOT NULL,
    category_id INTEGER REFERENCES categories(id),
    description TEXT
);

-- BẢNG KHÁCH HÀNG
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT
);

-- BẢNG ĐƠN HÀNG
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount NUMERIC(12, 0),
    status TEXT CHECK (status IN ('pending', 'shipped', 'cancelled')) DEFAULT 'pending'
);

-- BẢNG CHI TIẾT ĐƠN HÀNG
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price NUMERIC(12, 0) NOT NULL
);

-- DỮ LIỆU MẪU: DANH MỤC
INSERT INTO categories (name, description) VALUES
('Điện thoại', 'Các dòng smartphone phổ biến'),
('Thời trang', 'Quần áo, giày dép, phụ kiện'),
('Đồ gia dụng', 'Đồ dùng cho gia đình'),
('Laptop', 'Máy tính xách tay các loại'),
('Phụ kiện', 'Phụ kiện điện tử và công nghệ');

-- DỮ LIỆU MẪU: SẢN PHẨM
INSERT INTO products (name, price, stock, category_id, description) VALUES
('iPhone 15 Pro', 29990000, 20, 1, 'Smartphone cao cấp của Apple'),
('Samsung Galaxy S24', 24990000, 25, 1, 'Flagship Android mới nhất'),
('Áo thun nam', 199000, 100, 2, 'Áo cotton đơn giản, nhiều màu sắc'),
('Quần jean nữ', 450000, 50, 2, 'Quần jean co giãn thoải mái'),
('Máy hút bụi Xiaomi', 1500000, 15, 3, 'Hút bụi cầm tay không dây'),
('Nồi cơm điện', 899000, 30, 3, 'Nồi cơm 1.8L cho 4-6 người'),
('MacBook Air M2', 28990000, 10, 4, 'Laptop Apple chip M2 mới nhất'),
('Dell XPS 13', 25990000, 8, 4, 'Laptop Windows cao cấp'),
('Tai nghe AirPods Pro', 5990000, 40, 5, 'Tai nghe không dây chống ồn'),
('Chuột wireless Logitech', 790000, 60, 5, 'Chuột không dây gaming');

-- DỮ LIỆU MẪU: KHÁCH HÀNG
INSERT INTO customers (name, email, phone) VALUES
('Nguyễn Văn A', 'nguyenvana@email.com', '0901234567'),
('Trần Thị B', 'tranthib@email.com', '0912345678'),
('Lê Văn C', 'levanc@email.com', '0923456789'),
('Phạm Thị D', 'phamthid@email.com', '0934567890'),
('Hoàng Văn E', 'hoangvane@email.com', '0945678901');

-- DỮ LIỆU MẪU: ĐƠN HÀNG
INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(1, '2024-06-01 10:30:00', 30189000, 'shipped'),
(2, '2024-06-02 14:00:00', 199000, 'pending'),
(3, '2024-06-03 16:20:00', 26790000, 'shipped'),
(1, '2024-06-04 09:15:00', 6789000, 'pending'),
(4, '2024-06-05 11:45:00', 1349000, 'shipped'),
(5, '2024-06-06 13:30:00', 29889000, 'pending'),
(2, '2024-06-07 15:10:00', 1649000, 'cancelled'),
(3, '2024-06-08 08:25:00', 450000, 'shipped');

-- DỮ LIỆU MẪU: CHI TIẾT ĐƠN HÀNG
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
-- Đơn hàng 1: iPhone + Áo thun
(1, 1, 1, 29990000),
(1, 3, 1, 199000),

-- Đơn hàng 2: Áo thun
(2, 3, 1, 199000),

-- Đơn hàng 3: Samsung + Chuột
(3, 2, 1, 24990000),
(3, 10, 1, 790000),

-- Đơn hàng 4: AirPods + Chuột
(4, 9, 1, 5990000),
(4, 10, 1, 790000),

-- Đơn hàng 5: Máy hút bụi + Áo thun
(5, 5, 1, 1500000),
(5, 3, 1, 199000),

-- Đơn hàng 6: MacBook + Tai nghe
(6, 7, 1, 28990000),
(6, 9, 1, 5990000),

-- Đơn hàng 7: Máy hút bụi + Nồi cơm (đã hủy)
(7, 5, 1, 1500000),
(7, 6, 1, 899000),

-- Đơn hàng 8: Quần jean
(8, 4, 1, 450000);
"""
    
    try:
        # Kết nối database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        
        # Thực hiện script
        cursor.execute(sql_script)
        connection.commit()
        
        print("✅ Database đã được thiết lập thành công!")
        print("📊 Dữ liệu đã tạo:")
        print("   - 5 danh mục sản phẩm")
        print("   - 10 sản phẩm")
        print("   - 5 khách hàng")
        print("   - 8 đơn hàng")
        print("   - 13 chi tiết đơn hàng")
        
        # Kiểm tra dữ liệu
        cursor.execute("SELECT COUNT(*) FROM categories")
        categories_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM products")
        products_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM customers")
        customers_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM orders")
        orders_count = cursor.fetchone()[0]
        
        print(f"\n📈 Thống kê:")
        print(f"   - Danh mục: {categories_count}")
        print(f"   - Sản phẩm: {products_count}")
        print(f"   - Khách hàng: {customers_count}")
        print(f"   - Đơn hàng: {orders_count}")
        
    except Exception as e:
        print(f"❌ Lỗi thiết lập database: {e}")
        if connection:
            connection.rollback()
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    setup_database()