import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from bs4 import BeautifulSoup
import redis.asyncio as redis
import os
import signal
import sys
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
import aiohttp
import re
from urllib.parse import quote_plus
import html

# Install required packages:
# pip install redis google-generativeai dataclasses-json aiohttp beautifulsoup4 python-dotenv

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass_json
@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = None
    search_results: Optional[str] = None  # Store search results
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass_json
@dataclass
class ConversationContext:
    thread_id: str
    user_id: str
    messages: List[ChatMessage] = None
    metadata: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.metadata is None:
            self.metadata = {}
        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

class DuckDuckGoSearcher:
    """DuckDuckGo search functionality"""
    
    def __init__(self):
        self.session = None
        self.base_url = "https://html.duckduckgo.com/html/"
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search DuckDuckGo and return results"""
        try:
            if not self.session:
                await self.initialize()
            
            # Prepare search parameters
            params = {
                'q': query,
                'b': '',  # Start from first result
                'kl': 'wt-wt',  # No region restriction
                'df': '',  # No date filter
            }
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            async with self.session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"DuckDuckGo search failed with status: {response.status}")
                    return []
                
                html_content = await response.text()
                return self._parse_results(html_content, max_results)
                
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    def _parse_results(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Phân tích HTML từ DuckDuckGo và trích xuất kết quả tìm kiếm."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
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
                    
                    # Lấy snippet
                    desc_elem = container.find('a', class_='result__snippet')
                    snippet = desc_elem.get_text().strip() if desc_elem else ""
                    
                    # Fallback: thử với <div class="result__snippet">
                    if not snippet:
                        desc_elem = container.find('div', class_='result__snippet')
                        if desc_elem:
                            snippet = desc_elem.get_text().strip()
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })

                except Exception as e:
                    logger.warning(f"Lỗi khi phân tích kết quả đơn lẻ: {e}")
                    continue

            logger.info(f"✅ DuckDuckGo - Tìm được {len(results)} kết quả")
        
        except Exception as e:
            logger.error(f"❌ Lỗi khi phân tích HTML DuckDuckGo: {e}")
        
        return results
    
    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return "Không tìm thấy kết quả tìm kiếm."
        
        formatted = "=== KẾT QUẢ TÌM KIẾM ===\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   {result['snippet']}\n"
            if result['url'].startswith('http'):
                formatted += f"   URL: {result['url']}\n"
            formatted += "\n"
        
        formatted += "=== HẾT KẾT QUẢ TÌM KIẾM ===\n"
        return formatted

class RedisMemoryManager:
    """Redis memory manager for conversation history"""
    
    def __init__(self):
        self.redis_client = None
        self.redis_ttl = 3600  # 1 hour TTL for active sessions
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.aclose()
    
    def _get_redis_key(self, thread_id: str) -> str:
        """Generate Redis key for thread"""
        return f"chat:thread:{thread_id}"
    
    async def get_conversation(self, thread_id: str) -> Optional[ConversationContext]:
        """Get conversation from Redis"""
        try:
            redis_key = self._get_redis_key(thread_id)
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                logger.info(f"Cache HIT for thread {thread_id}")
                data = json.loads(cached_data)
                
                # Convert message dicts back to ChatMessage objects
                if 'messages' in data and data['messages']:
                    messages = []
                    for msg_data in data['messages']:
                        if isinstance(msg_data, dict):
                            messages.append(ChatMessage(**msg_data))
                        else:
                            messages.append(msg_data)
                    data['messages'] = messages
                
                return ConversationContext(**data)
            
            logger.info(f"Cache MISS for thread {thread_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting conversation {thread_id}: {e}")
            return None
    
    async def save_conversation(self, conversation: ConversationContext):
        """Save conversation to Redis"""
        try:
            conversation.updated_at = datetime.now().isoformat()
            
            # Convert to dict for JSON serialization
            data = asdict(conversation)
            
            redis_key = self._get_redis_key(conversation.thread_id)
            await self.redis_client.setex(
                redis_key,
                self.redis_ttl,
                json.dumps(data, default=str)
            )
            
            logger.info(f"Conversation {conversation.thread_id} saved to Redis")
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation.thread_id}: {e}")
            raise

class HistoryProcessor:
    """Xử lý lịch sử và tạo query search thông minh"""
    
    def __init__(self, model):
        self.model = model
    
    async def process_history_and_input(self, messages: List[ChatMessage], current_input: str) -> Dict[str, Any]:
        """
        Xử lý lịch sử chat và input hiện tại để:
        1. Tóm tắt ngữ cảnh quan trọng
        2. Xác định có cần search không
        3. Tạo query search tối ưu
        """
        print("🧠 ĐANG XỬ LÝ LỊCH SỬ VÀ INPUT...")
        
        # Lấy lịch sử gần đây (10 tin nhắn cuối)
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        # Tạo summary của lịch sử
        history_summary = self._create_history_summary(recent_messages)
        
        # Phân tích input hiện tại
        input_analysis = await self._analyze_current_input(current_input, history_summary)
        
        # Quyết định có cần search không
        need_search = input_analysis.get('need_search', False)
        
        # Tạo search query nếu cần
        search_query = None
        if need_search:
            search_query = input_analysis.get('search_query', current_input)
        
        result = {
            'history_summary': history_summary,
            'input_analysis': input_analysis,
            'need_search': need_search,
            'search_query': search_query,
            'context_for_llm': self._create_context_for_llm(history_summary, current_input)
        }
        
        print(f"📋 KẾT QUẢ XỬ LÝ:")
        print(f"   - Tóm tắt lịch sử: {len(history_summary)} ký tự")
        print(f"   - Cần search: {'CÓ' if need_search else 'KHÔNG'}")
        if search_query:
            print(f"   - Query search: '{search_query}'")
        print()
        
        return result
    
    def _create_history_summary(self, messages: List[ChatMessage]) -> str:
        """Tạo tóm tắt lịch sử chat"""
        if not messages:
            return "Chưa có lịch sử trò chuyện."
        
        # Tóm tắt các chủ đề chính
        topics = []
        entities = []
        
        for msg in messages:
            content = msg.content
            
            # Trích xuất thực thể quan trọng (tên người, công ty, địa điểm)
            # Pattern cho tên người Việt
            name_pattern = r'([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+ [A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)'
            names = re.findall(name_pattern, content)
            entities.extend(names)
            
            # Pattern cho công ty/tổ chức
            company_pattern = r'((?:CÔNG TY|CT|CÔNG TY TNHH|TNHH|JSC|Co\.|Ltd\.|GROUP|THACO|VINGROUP|FPT|VIETTEL)[^.!?\n]*)'
            companies = re.findall(company_pattern, content, re.IGNORECASE)
            entities.extend(companies)
        
        # Loại bỏ trùng lặp và làm sạch
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 3]))
        
        # Tạo summary
        summary_parts = []
        
        if entities:
            summary_parts.append(f"Thực thể đã đề cập: {', '.join(entities[:5])}")  # Lấy 5 thực thể đầu
        
        # Tóm tắt nội dung chính từ 3 tin nhắn gần nhất
        recent_content = []
        for msg in messages[-3:]:
            if msg.role == 'user':
                recent_content.append(f"User hỏi: {msg.content[:100]}")
            else:
                recent_content.append(f"Bot trả lời về: {msg.content[:100]}")
        
        if recent_content:
            summary_parts.append("Nội dung gần đây: " + "; ".join(recent_content))
        
        return ". ".join(summary_parts) if summary_parts else "Cuộc trò chuyện chung."
    
    async def _analyze_current_input(self, current_input: str, history_summary: str) -> Dict[str, Any]:
        """Phân tích input hiện tại với ngữ cảnh lịch sử"""
        
        # Prompt để phân tích input
        analysis_prompt = f"""
Hãy phân tích câu hỏi sau với ngữ cảnh lịch sử trò chuyện:

LỊCH SỬ TRÒ CHUYỆN: {history_summary}

CÂU HỎI HIỆN TẠI: {current_input}

Hãy trả lời theo format JSON:
{{
    "need_search": true/false,
    "reason": "lý do cần/không cần search",
    "search_query": "query tìm kiếm tối ưu (nếu cần)",
    "resolved_entities": ["danh sách thực thể đã được giải quyết từ lịch sử"],
    "search_intent": "ý định tìm kiếm"
}}

Quy tắc quyết định:
1. CÓ cần search nếu: hỏi về thông tin cập nhật, tin tức, giá cả, thời tiết, sự kiện gần đây
2. KHÔNG cần search nếu: câu hỏi chung, giải thích khái niệm, hỏi về lịch sử, toán học
3. Khi tạo search query, hãy thay thế đại từ (ông ấy, bà ấy, anh ta...) bằng tên cụ thể từ lịch sử
4. Ưu tiên từ khóa quan trọng, loại bỏ từ dừng không cần thiết
"""
        
        try:
            # Gửi request tới Gemini để phân tích
            response = await asyncio.to_thread(
                self.model.generate_content, 
                analysis_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=500
                )
            )
            
            # Parse JSON response
            analysis_text = response.text.strip()
            
            # Tìm JSON trong response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group())
                print(f"🤖 PHÂN TÍCH TỪ LLM: {analysis_json}")
                return analysis_json
            else:
                print("⚠️ Không parse được JSON, dùng fallback logic")
                return self._fallback_analysis(current_input)
                
        except Exception as e:
            print(f"❌ Lỗi phân tích LLM: {e}, dùng fallback logic")
            return self._fallback_analysis(current_input)
    
    def _fallback_analysis(self, current_input: str) -> Dict[str, Any]:
        """Logic fallback khi LLM analysis fail"""
        # Keywords suggest search is needed
        search_keywords = [
            'tin tức', 'news', 'mới nhất', 'hiện tại', 'cập nhật',
            'thời tiết', 'weather', 'giá', 'price', 'tỷ giá',
            'sự kiện', 'event', 'lịch', 'schedule', 'thông tin',
            'hôm nay', 'today', 'bây giờ', 'now', 'real-time'
        ]
        
        input_lower = current_input.lower()
        found_keywords = [kw for kw in search_keywords if kw in input_lower]
        need_search = len(found_keywords) > 0
        
        return {
            'need_search': need_search,
            'reason': f"Phát hiện từ khóa: {', '.join(found_keywords)}" if need_search else "Không có từ khóa cần search",
            'search_query': current_input if need_search else None,
            'resolved_entities': [],
            'search_intent': 'general_search' if need_search else 'knowledge_query'
        }
    
    def _create_context_for_llm(self, history_summary: str, current_input: str) -> str:
        """Tạo context tổng hợp để gửi cho LLM chính"""
        return f"""
NGỮ CẢNH CUỘC TRÒ CHUYỆN:
{history_summary}

CÂU HỎI HIỆN TẠI:
{current_input}
"""

class GeminiChatbot:
    """Gemini Flash chatbot with Redis memory and intelligent search"""
    
    def __init__(self, memory_manager: RedisMemoryManager, searcher: DuckDuckGoSearcher):
        self.memory = memory_manager
        self.searcher = searcher
        self.model = None
        
        # Initialize Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize history processor
        self.history_processor = HistoryProcessor(self.model)
        
        logger.info("Gemini Flash model initialized")
    
    async def chat(self, user_id: str, thread_id: str, user_message: str) -> str:
        """Process chat message with intelligent history-aware search"""
        try:
            print("\n" + "="*80)
            print("🔄 BẮT ĐẦU XỬ LÝ CHAT")
            print("="*80)
            
            # Get or create conversation
            conversation = await self.memory.get_conversation(thread_id)
            
            if not conversation:
                conversation = ConversationContext(
                    thread_id=thread_id,
                    user_id=user_id,
                    messages=[],
                    metadata={"model": "gemini-2.0-flash", "search_enabled": True}
                )
            
            # Print user input
            print(f"📝 USER INPUT:")
            print(f"   {user_message}")
            print()
            
            # Print conversation history
            print(f"📚 LỊCH SỬ HỘI THOẠI ({len(conversation.messages)} tin nhắn):")
            if conversation.messages:
                for i, msg in enumerate(conversation.messages[-5:], 1):  # Show last 5 messages
                    role_icon = "👤" if msg.role == "user" else "🤖"
                    print(f"   {i}. {role_icon} {msg.role}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            else:
                print("   (Chưa có lịch sử)")
            print()
            
            # STEP 1: Process history and current input
            processing_result = await self.history_processor.process_history_and_input(
                conversation.messages, user_message
            )
            
            # STEP 2: Perform search if needed
            search_results_text = ""
            if processing_result['need_search']:
                search_query = processing_result['search_query']
                print(f"🔍 ĐANG TÌM KIẾM VỚI QUERY: '{search_query}'")
                
                search_results = await self.searcher.search(search_query, max_results=5)
                search_results_text = self.searcher.format_search_results(search_results)
                
                print("📊 KẾT QUẢ TÌM KIẾM:")
                if search_results:
                    for i, result in enumerate(search_results, 1):
                        print(f"   {i}. {result['title']}")
                        print(f"      Snippet: {result['snippet'][:150]}{'...' if len(result['snippet']) > 150 else ''}")
                        print()
                else:
                    print("   Không tìm thấy kết quả")
                print()
            
            # STEP 3: Create enhanced prompt
            enhanced_message = self._create_enhanced_prompt(
                processing_result['context_for_llm'],
                user_message,
                search_results_text
            )
            
            # Print final prompt to LLM
            print("🤖 PROMPT ĐƯA VÀO GEMINI:")
            print("-" * 60)
            print(enhanced_message[:500] + "..." if len(enhanced_message) > 500 else enhanced_message)
            print("-" * 60)
            print()
            
            # Add user message to conversation
            user_msg = ChatMessage(
                role="user", 
                content=user_message,
                search_results=search_results_text if search_results_text else None
            )
            conversation.messages.append(user_msg)
            
            # Prepare conversation history for Gemini
            history = self._prepare_gemini_history(conversation.messages[:-1])
            
            print("⏳ ĐANG GỬI REQUEST ĐẾN GEMINI...")
            
            # Start chat with history
            chat = self.model.start_chat(history=history)
            
            # Send enhanced message and get response
            response = await asyncio.to_thread(chat.send_message, enhanced_message)
            bot_response = response.text
            
            print("✅ NHẬN ĐƯỢC PHẢN HỒI TỪ GEMINI:")
            print("-" * 60)
            print(bot_response[:300] + "..." if len(bot_response) > 300 else bot_response)
            print("-" * 60)
            print()
            
            # Add bot message
            bot_msg = ChatMessage(role="assistant", content=bot_response)
            conversation.messages.append(bot_msg)
            
            # Keep only last 20 messages to prevent memory bloat
            if len(conversation.messages) > 20:
                conversation.messages = conversation.messages[-20:]
                print("🗑️ ĐÃ XÓA TIN NHẮN CŨ (GIỮ 20 TIN NHẮN GẦN NHẤT)")
            
            # Save conversation
            await self.memory.save_conversation(conversation)
            print("💾 ĐÃ LUU CONVERSATION VÀO REDIS")
            
            print("="*80)
            print("✨ HOÀN THÀNH XỬ LÝ CHAT")
            print("="*80)
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"Xin lỗi, tôi gặp lỗi: {str(e)}"
    
    def _create_enhanced_prompt(self, context: str, user_message: str, search_results: str) -> str:
        """Tạo prompt tổng hợp cho LLM"""
        prompt_parts = []
        
        # Thêm context
        prompt_parts.append(f"NGỮ CẢNH CUỘC TRÒ CHUYỆN:\n{context}")
        
        # Thêm search results nếu có
        if search_results:
            prompt_parts.append(f"\nTHÔNG TIN TÌM KIẾM:\n{search_results}")
        
        # Thêm instruction
        if search_results:
            instruction = """
HƯỚNG DẪN TRẢ LỜI:
- Ưu tiên sử dụng thông tin từ kết quả tìm kiếm nếu liên quan
- Kết hợp với ngữ cảnh cuộc trò chuyện để trả lời chính xác
- Nếu thông tin tìm kiếm không đủ, hãy nói rõ và dùng kiến thức của bạn
- Trả lời bằng tiếng Việt, tự nhiên và thân thiện
"""
        else:
            instruction = """
HƯỚNG DẪN TRẢ LỜI:
- Sử dụng ngữ cảnh cuộc trò chuyện để trả lời chính xác
- Trả lời dựa trên kiến thức của bạn
- Trả lời bằng tiếng Việt, tự nhiên và thân thiện
"""
        
        prompt_parts.append(instruction)
        prompt_parts.append(f"\nCÂU HỎI: {user_message}")
        
        return "\n".join(prompt_parts)
    
    def _prepare_gemini_history(self, messages: List[ChatMessage]) -> List[Dict]:
        """Convert messages to Gemini chat history format"""
        history = []
        
        for msg in messages:
            # Gemini uses 'user' and 'model' roles
            role = 'user' if msg.role == 'user' else 'model'
            history.append({
                'role': role,
                'parts': [msg.content]
            })
        
        return history

class ChatbotConsole:
    """Console interface for the chatbot"""
    
    def __init__(self):
        self.memory_manager = None
        self.searcher = None
        self.chatbot = None
        self.running = True
        self.thread_id = str(uuid.uuid4())
        self.user_id = "console_user"
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n👋 Tạm biệt! Đang thoát...")
        self.running = False
    
    async def initialize(self):
        """Initialize chatbot services"""
        try:
            # Setup signal handler for Ctrl+C
            signal.signal(signal.SIGINT, self.signal_handler)
            
            # Initialize memory manager
            self.memory_manager = RedisMemoryManager()
            await self.memory_manager.initialize()
            
            # Initialize searcher
            self.searcher = DuckDuckGoSearcher()
            await self.searcher.initialize()
            
            # Initialize chatbot
            self.chatbot = GeminiChatbot(self.memory_manager, self.searcher)
            
            logger.info("Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.memory_manager:
            await self.memory_manager.close()
        if self.searcher:
            await self.searcher.close()
    
    def print_welcome(self):
        """Print welcome message"""
        print("=" * 70)
        print("🤖 GEMINI FLASH CHATBOT với REDIS MEMORY & DUCKDUCKGO SEARCH")
        print("🚀 NÂNG CẤP: Tìm kiếm có ngữ cảnh (Context-Aware Search)")
        print("=" * 70)
        print("💡 Hướng dẫn:")
        print("   • Nhập tin nhắn và nhấn ENTER để gửi")
        print("   • Nhấn Ctrl+C để thoát")
        print("   • Bot sẽ tự động tìm kiếm thông tin khi cần thiết")
        print("   • Bot sẽ nhớ lịch sử trò chuyện và sử dụng ngữ cảnh cho tìm kiếm")
        print(f"   • Thread ID: {self.thread_id}")
        print("=" * 70)
        print()
    
    async def run(self):
        """Main chat loop"""
        try:
            await self.initialize()
            self.print_welcome()
            
            while self.running:
                try:
                    # Get user input
                    user_input = input("👤 Bạn: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if not self.running:
                        break
                    
                    # Get bot response (detailed processing will be printed inside)
                    response = await self.chatbot.chat(
                        self.user_id, 
                        self.thread_id, 
                        user_input
                    )
                    
                    # Show final response
                    print(f"🎯 PHẢN HỒI CUỐI CÙNG CHO NGƯỜI DÙNG:")
                    print(f"🤖 Bot: {response}")
                    print()
                    
                except EOFError:
                    # Handle Ctrl+D
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    break
                except Exception as e:
                    print(f"\r❌ Lỗi: {e}")
                    print()
            
        except KeyboardInterrupt:
            pass
        finally:
            await self.cleanup()
            print("Cảm ơn bạn đã sử dụng chatbot! 👋")

# Demo function
async def demo():
    """Demo function to test the chatbot"""
    console = ChatbotConsole()
    await console.run()

if __name__ == "__main__":
    # Create .env file or set environment variables:
    """
    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_PASSWORD=
    GEMINI_API_KEY=your_gemini_api_key_here
    """
    
    print("Khởi động chatbot với tính năng tìm kiếm...")
    
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nThoát chatbot.")
    except Exception as e:
        print(f"Lỗi: {e}")
        sys.exit(1)