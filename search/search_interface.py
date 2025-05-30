import re
import time
import random
from typing import List
from datetime import datetime
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup
from .models import SearchResult, RankedResult
from .reranker import ContentReranker
from .gemini_api import GeminiAPI

class SearchInterface:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.reranker = ContentReranker()
        self.search_history = []
        self.gemini_api = GeminiAPI(api_key, model)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.cache = {}
        # Thêm thông tin thời gian
        self.current_date = datetime.now().strftime("%d/%m/%Y")
        self.llm_training_cutoff = "tháng 6/2024"  # Thời điểm model được train đến

    def search_bing(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Tìm kiếm kết quả từ Bing (ổn định và dễ scrape)."""
        retries = 3
        backoff_factor = 5

        for attempt in range(retries):
            try:
                time.sleep(random.uniform(2, 5))
                encoded_query = quote_plus(query)
                search_url = f"https://www.bing.com/search?q={encoded_query}&setLang=vi"

                print(f"🌐 Đang tìm kiếm trên Bing: {query}")

                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                }

                response = requests.get(search_url, headers=headers, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                results = []

                result_blocks = soup.select("li.b_algo")

                for block in result_blocks[:max_results]:
                    try:
                        title_elem = block.find("h2")
                        link_elem = title_elem.find("a") if title_elem else None
                        desc_elem = block.find("p")

                        if not link_elem or not link_elem.get("href", "").startswith("http"):
                            continue

                        title = title_elem.get_text().strip()
                        url = link_elem["href"]
                        description = desc_elem.get_text().strip() if desc_elem else ""

                        content = self.scrape_content(url)

                        results.append(SearchResult(
                            title=title,
                            url=url,
                            description=description,
                            content=content,
                            source="Bing"
                        ))
                        print(f"✅ Bing - Tìm thấy: {title}")

                    except Exception as e:
                        print(f"⚠️ Lỗi parse Bing result: {e}")
                        continue

                print(f"🔍 Bing hoàn thành: {len(results)} kết quả")
                return results

            except Exception as e:
                print(f"❌ Lỗi Bing search: {e}")
                time.sleep(backoff_factor * (2 ** attempt))

        print("❌ Bing search thất bại sau nhiều lần thử.")
        return []



    def detect_time_sensitive_query(self, query: str) -> bool:
        """Phát hiện câu hỏi liên quan đến thời gian hiện tại."""
        time_keywords = [
            'tuổi', 'age', 'năm nay', 'hiện tại', 'bây giờ', 'hôm nay', 'today', 'now', 'current',
            'mới nhất', 'latest', 'recent', 'gần đây', '2024', '2025', 'this year',
            'tin tức', 'news', 'cập nhật', 'update', 'thời sự'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in time_keywords)

    def add_time_context_to_prompt(self, original_prompt: str, query: str) -> str:
        """Thêm context thời gian vào prompt cho LLM."""
        if self.detect_time_sensitive_query(query):
            time_context = f"""
            QUAN TRỌNG - THÔNG TIN THỜI GIAN:
            - Ngày hiện tại: {self.current_date} (30/5/2025)
            - Model được train đến: {self.llm_training_cutoff} (tháng 6/2024)
            - Khi trả lời câu hỏi về tuổi, thời gian, sự kiện hiện tại, hãy sử dụng thông tin từ search results để cập nhật đến thời điểm hiện tại (2025)
            - Nếu không có thông tin mới từ search, hãy nói rõ rằng thông tin có thể đã lỗi thời

            """
            return time_context + original_prompt
        
        return original_prompt

    def search_duckduckgo(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Tìm kiếm thực tế với DuckDuckGo (web scraping)."""
        retries = 3
        backoff_factor = 5
        
        for attempt in range(retries):
            try:
                # Thêm delay ngẫu nhiên để tránh bị block
                time.sleep(random.uniform(2, 5))
                
                # Encode query cho URL
                encoded_query = quote_plus(query)
                search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
                
                print(f"🌐 Đang tìm kiếm trên DuckDuckGo: {query}")
                
                # Gửi yêu cầu với các tham số phù hợp
                response = self.session.get(search_url, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Tìm các kết quả tìm kiếm DuckDuckGo
                search_containers = soup.find_all('div', class_='result__body')
                
                for container in search_containers[:max_results]:
                    try:
                        # Lấy title
                        title_elem = container.find('a', class_='result__a')
                        if not title_elem:
                            continue
                        title = title_elem.get_text().strip()
                        
                        # Lấy URL
                        url = title_elem.get('href')
                        if not url or not url.startswith('http'):
                            continue
                        
                        # Lấy description (snippet)
                        desc_elem = container.find('a', class_='result__snippet')
                        description = desc_elem.get_text().strip() if desc_elem else ""
                        
                        if not description:
                            # Fallback: tìm mô tả trong thẻ khác
                            desc_elem = container.find('div', class_='result__snippet')
                            if desc_elem:
                                description = desc_elem.get_text().strip()
                        
                        if title and url:
                            # Scrape nội dung từ URL
                            content = self.scrape_content(url)
                            
                            search_result = SearchResult(
                                title=title,
                                url=url,
                                description=description,
                                content=content,
                                source="DuckDuckGo"
                            )
                            results.append(search_result)
                            
                            print(f"✅ DuckDuckGo - Tìm thấy: {title}")
                    
                    except Exception as e:
                        print(f"⚠️ Lỗi parse DuckDuckGo result: {e}")
                        continue
                
                print(f"🔍 DuckDuckGo hoàn thành: {len(results)} kết quả")
                return results
            
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    print(f"⚠️ DuckDuckGo 429: Quá nhiều yêu cầu. Thử lại sau {backoff_factor * (2 ** attempt)} giây...")
                    time.sleep(backoff_factor * (2 ** attempt))
                else:
                    print(f"❌ Lỗi DuckDuckGo search: {e}")
                    return []
            except Exception as e:
                print(f"❌ Lỗi DuckDuckGo search: {e}")
                return []
    
        print("❌ DuckDuckGo search thất bại sau nhiều lần thử.")
        return []

    def search_combined(self, query: str, total_results: int = 10) -> List[SearchResult]:
        """Kết hợp tìm kiếm từ Bing và DuckDuckGo."""
        print(f"\n🔍 Bắt đầu tìm kiếm kết hợp: '{query}'")
        print("📊 Chiến lược: Bing (5 kết quả) + DuckDuckGo (5 kết quả)")
        
        # Chia đều kết quả giữa 2 search engine
        results_per_engine = total_results // 2
        
        all_results = []
        
        # Tìm kiếm song song (có thể tối ưu với threading sau)
        ping_results = self.search_bing(query, max_results=results_per_engine)
        duckduckgo_results = self.search_duckduckgo(query, max_results=results_per_engine)
        
        # Kết hợp kết quả
        all_results.extend(ping_results)
        all_results.extend(duckduckgo_results)
        
        # Loại bỏ duplicate URLs
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        print(f"🎯 Tổng kết: {len(unique_results)} kết quả unique từ {len(all_results)} kết quả gốc")
        
        return unique_results[:total_results]  # Giới hạn theo yêu cầu

    def scrape_content(self, url: str, max_length: int = 1500) -> str:
        """Scrape nội dung từ URL."""
        try:
            # Thêm delay nhỏ
            time.sleep(random.uniform(0.5, 1.5))
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Loại bỏ script, style, nav, footer
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                
                # Tìm nội dung chính
                content_selectors = [
                    'article', '.content', '.main-content', '.post-content', 
                    '.entry-content', '#content', 'main', '.article-body', 
                    '.story-body', '.post-body', '.content-body'
                ]
                
                content = ""
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content = content_elem.get_text(separator=' ', strip=True)
                        break
                
                # Nếu không tìm thấy, lấy từ body
                if not content:
                    body = soup.find('body')
                    if body:
                        content = body.get_text(separator=' ', strip=True)
                
                # Làm sạch và giới hạn độ dài
                if content:
                    content = re.sub(r'\s+', ' ', content).strip()
                    return content[:max_length] + "..." if len(content) > max_length else content
            
        except Exception as e:
            print(f"⚠️ Lỗi scrape {url}: {e}")
        
        return ""

    def start_interactive_search(self):
        """Bắt đầu phiên tìm kiếm tương tác."""
        print("=" * 60)
        print("🔍 HỆ THỐNG TÌM KIẾM THÔNG MINH VỚI GEMINI AI")
        print("=" * 60)
        print("💡 Chỉ cần nhập câu hỏi, hệ thống sẽ tự động:")
        print("   • Tìm kiếm trên Bing + DuckDuckGo")
        print("   • Kết hợp và loại bỏ trùng lặp")
        print("   • Sắp xếp kết quả theo độ liên quan với Reranker")
        print("   • Tạo câu trả lời thông minh với Gemini AI")
        print("=" * 60)
        print("Nhập 'quit' để thoát | 'history' để xem lịch sử | 'help' để xem hướng dẫn")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n💬 Hỏi gì đó: ").strip()
                
                if not query:
                    print("⚠️ Vui lòng nhập câu hỏi!")
                    continue
                
                if query.lower() in ['quit', 'exit', 'thoát']:
                    print("👋 Cảm ơn bạn đã sử dụng! Tạm biệt!")
                    break
                
                elif query.lower() in ['history', 'lịch sử']:
                    self.show_search_history()
                    continue
                
                elif query.lower() in ['help', 'hướng dẫn']:
                    self.show_help()
                    continue
                
                # Thực hiện tìm kiếm và trả lời
                self.perform_search(query)
                
                continue_choice = input("\n❓ Tiếp tục hỏi? (Enter để tiếp tục, 'q' để thoát): ").strip()
                if continue_choice.lower() in ['q', 'quit', 'thoát']:
                    print("👋 Cảm ơn bạn đã sử dụng!")
                    break
                
            except KeyboardInterrupt:
                print("\n👋 Cảm ơn bạn đã sử dụng! Tạm biệt!")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                print("🔄 Vui lòng thử lại...")

    def perform_search(self, query: str, total_results: int = 10, top_k: int = 5):
        """Thực hiện tìm kiếm kết hợp với query."""
        print(f"\n🔍 Đang tìm kiếm: '{query}'")
        print("⏳ Vui lòng đợi...")
        
        # Lưu vào lịch sử
        self.search_history.append({
            'query': query,
            'timestamp': self.get_current_time()
        })
        
        # Tìm kiếm kết hợp Bing + DuckDuckGo
        combined_results = self.search_combined(query, total_results=total_results)
        
        if not combined_results:
            print("❌ Không tìm thấy kết quả nào! Vui lòng thử lại sau.")
            return
            
        print(f"✅ Tổng cộng tìm thấy {len(combined_results)} kết quả từ cả Bing và DuckDuckGo")
        
        # Rerank với hybrid method
        print("🔄 Đang sắp xếp lại kết quả theo độ liên quan với Reranker...")
        ranked_results = self.reranker.rerank_hybrid(query, combined_results, top_k)
        
        # Tạo context từ kết quả đã rerank
        search_context = self._format_search_context(ranked_results)
        
        # Gọi Gemini API để tạo câu trả lời
        print("🤖 Đang tạo câu trả lời với Gemini AI...")
        answer = self._generate_answer_with_gemini(query, search_context)
        
        # Hiển thị kết quả cuối cùng
        self.display_final_answer(query, answer, ranked_results)

    def _format_search_context(self, ranked_results: List[RankedResult]) -> str:
        """Format search context để gửi cho Gemini."""
        if not ranked_results:
            return "Không tìm thấy thông tin liên quan."
        
        context = "THÔNG TIN TÌM KIẾM (ĐÃ SẮP XẾP THEO ĐỘ LIÊN QUAN - BING + DUCKDUCKGO):\n\n"
        
        for i, ranked in enumerate(ranked_results, 1):
            result = ranked.original_result
            context += f"[Nguồn {i} - {result.source}] {result.title}\n"
            context += f"URL: {result.url}\n"
            context += f"Mô tả: {result.description}\n"
            
            if result.content:
                content = result.content[:800] + "..." if len(result.content) > 800 else result.content
                context += f"Nội dung: {content}\n"
            
            context += f"Điểm liên quan: {ranked.combined_score:.2f}\n"
            context += "\n" + "-"*50 + "\n\n"
        
        return context

    def _generate_answer_with_gemini(self, query: str, search_context: str) -> str:
        """Gọi Gemini API để tạo câu trả lời với time context."""
        base_prompt = f"""
Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi dựa trên thông tin tìm kiếm được cung cấp từ Bing và DuckDuckGo.

CÂU HỎI: {query}

{search_context}

HƯỚNG DẪN TRẢ LỜI:
1. Trả lời trực tiếp và đầy đủ câu hỏi
2. Sử dụng thông tin từ các nguồn đáng tin cậy (ưu tiên nguồn có điểm cao)
3. Tổng hợp thông tin từ nhiều nguồn Bing và DuckDuckGo để đưa ra câu trả lời toàn diện
4. Đề cập nguồn thông tin khi cần thiết (ví dụ: "Theo nguồn 1..." hoặc "Các nghiên cứu cho thấy...")
5. Nếu có thông tin mâu thuẫn giữa các nguồn, hãy chỉ ra và đưa ra quan điểm cân bằng
6. Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
7. Cấu trúc câu trả lời một cách logic và có tổ chức
8. Nếu không có đủ thông tin để trả lời đầy đủ, hãy nói rõ điều này

Câu trả lời:
"""
        
        # Thêm time context nếu cần
        enhanced_prompt = self.add_time_context_to_prompt(base_prompt, query)
        
        return self.gemini_api.generate_answer(enhanced_prompt)

    def display_final_answer(self, query: str, answer: str, ranked_results: List[RankedResult]):
        """Hiển thị câu trả lời cuối cùng."""
        print("\n" + "="*70)
        print("🎯 CÂU TRẢ LỜI CHO CÂU HỎI CỦA BẠN")
        print("="*70)
        print(f"❓ Câu hỏi: {query}")
        print("-"*70)
        print("🤖 Trả lời từ Gemini AI:")
        print(answer)
        print("-"*70)
        
        # Hiển thị nguồn tham khảo với thông tin search engine
        print("📚 Nguồn tham khảo (Bing + DuckDuckGo):")
        for i, ranked in enumerate(ranked_results, 1):
            result = ranked.original_result
            print(f"  {i}. [{result.source}] {result.title}")
            print(f"     {result.url}")
            print(f"     Độ tin cậy: {ranked.combined_score:.1%}")
        
        print("="*70)

    def show_search_history(self):
        """Hiển thị lịch sử tìm kiếm."""
        if not self.search_history:
            print("📝 Chưa có lịch sử tìm kiếm nào!")
            return
        
        print("\n" + "="*50)
        print("📚 LỊCH SỬ TÌM KIẾM")
        print("="*50)
        
        for i, item in enumerate(self.search_history[-10:], 1):
            print(f"{i}. '{item['query']}' - {item['timestamp']}")

    def show_help(self):
        """Hiển thị hướng dẫn sử dụng."""
        print("\n" + "="*50)
        print("📖 HƯỚNG DẪN SỬ DỤNG")
        print("="*50)
        print("🔹 Nhập câu hỏi bất kỳ để được trả lời bởi Gemini AI")
        print("🔹 Nhập 'quit' hoặc 'thoát' để thoát")
        print("🔹 Nhập 'history' hoặc 'lịch sử' để xem lịch sử")
        print("🔹 Nhập 'help' hoặc 'hướng dẫn' để xem hướng dẫn này")
        print("\n⚙️ Hệ thống tự động:")
        print("• Tìm kiếm kết hợp trên Bing + DuckDuckGo (mỗi engine 5 kết quả)")
        print("• Loại bỏ URL trùng lặp giữa các search engine")
        print("• Scrape nội dung từ các trang web")
        print("• Sắp xếp kết quả theo độ liên quan (Hybrid Reranking)")
        print("• Tạo câu trả lời thông minh với Gemini AI")
        print("\n💡 Ví dụ câu hỏi:")
        print("• 'Python là gì và tại sao nên học?'")
        print("• 'Cách học machine learning hiệu quả'")
        print("• 'Xu hướng công nghệ AI 2024'")
        print("• 'Lợi ích của việc tập thể dục'")
        print("\n⚠️ Lưu ý:")
        print("• Hệ thống sử dụng web scraping, có thể chậm hơn API")
        print("• Tránh tìm kiếm quá nhiều lần liên tiếp để không bị chặn")
        print("• Kết quả được kết hợp từ cả Bing và DuckDuckGo")

    def get_current_time(self) -> str:
        """Lấy thời gian hiện tại."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def search_with_time_awareness(self, query: str, total_results: int = 10, top_k: int = 5):
        """Tìm kiếm với khả năng nhận biết thời gian."""
        print(f"\n🔍 Đang phân tích câu hỏi: '{query}'")
        
        # Kiểm tra nếu câu hỏi liên quan đến thời gian
        if self.detect_time_sensitive_query(query):
            print("⏰ Phát hiện câu hỏi liên quan thời gian - sẽ ưu tiên thông tin mới nhất")
            # Thêm từ khóa để tìm thông tin mới
            enhanced_query = f"{query} 2024 2025 latest current"
        else:
            enhanced_query = query
        
        # Thực hiện tìm kiếm
        self.perform_search(enhanced_query, total_results, top_k)