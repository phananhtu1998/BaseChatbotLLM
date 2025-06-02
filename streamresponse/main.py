# main.py - Backend FastAPI với Gemini 2.0 Flash
# requirements.txt:
# fastapi==0.104.1
# uvicorn==0.24.0
# google-generativeai==0.3.2
# python-multipart==0.0.6
# pydantic==2.5.0

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import json
import asyncio
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    role: str  # 'user' hoặc 'assistant'
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    model: Optional[str] = "gemini-2.0-flash-exp"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: datetime

# FastAPI app
app = FastAPI(
    title="Gemini 2.0 Flash Chatbot API",
    description="API cho chatbot sử dụng Gemini 2.0 Flash với streaming response",
    version="1.0.0"
)

# CORS middleware - cho phép frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (để serve HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY chưa được set. Hãy set trong environment variables.")
    logger.info("💡 Cách set: export GEMINI_API_KEY='your-api-key'")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class GeminiChatbot:
    def __init__(self):
        self.model_name = "gemini-2.0-flash-exp"
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

    def get_model(self):
        """Tạo model Gemini với cấu hình"""
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

    async def stream_response(self, message: str, history: List[ChatMessage] = None):
        """Stream response từ Gemini với real-time chunks"""
        try:
            model = self.get_model()
            
            # Chuyển đổi history sang format Gemini
            gemini_history = []
            if history:
                for msg in history[-10]:  # Giữ 10 tin nhắn gần nhất để tránh context quá dài
                    role = "user" if msg.role == "user" else "model"
                    gemini_history.append({
                        "role": role,
                        "parts": [{"text": msg.content}]
                    })

            # Tạo chat session với history
            chat = model.start_chat(history=gemini_history)
            
            # Gọi Gemini API với streaming
            response = await asyncio.to_thread(
                chat.send_message, 
                message, 
                stream=True
            )
            
            full_response = ""
            chunk_count = 0
            
            # Stream từng chunk
            for chunk in response:
                if chunk.text:
                    chunk_text = chunk.text
                    full_response += chunk_text
                    chunk_count += 1
                    
                    # Gửi chunk data qua Server-Sent Events
                    yield f"data: {json.dumps({
                        'type': 'chunk',
                        'content': chunk_text,
                        'full_content': full_response,
                        'chunk_id': chunk_count,
                        'timestamp': datetime.now().isoformat()
                    }, ensure_ascii=False)}\n\n"
                    
                    # Thêm delay nhỏ để tạo hiệu ứng streaming tự nhiên
                    await asyncio.sleep(0.02)
            
            # Gửi signal hoàn thành
            yield f"data: {json.dumps({
                'type': 'done',
                'content': '',
                'full_content': full_response,
                'total_chunks': chunk_count,
                'timestamp': datetime.now().isoformat()
            }, ensure_ascii=False)}\n\n"
            
            logger.info(f"✅ Stream completed: {chunk_count} chunks, {len(full_response)} chars")
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {str(e)}")
            yield f"data: {json.dumps({
                'type': 'error',
                'error': f'Lỗi khi kết nối với Gemini: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }, ensure_ascii=False)}\n\n"

    async def get_response(self, message: str, history: List[ChatMessage] = None):
        """Non-streaming response (backup method)"""
        try:
            model = self.get_model()
            
            # Chuyển đổi history
            gemini_history = []
            if history:
                for msg in history[-10]:
                    role = "user" if msg.role == "user" else "model"
                    gemini_history.append({
                        "role": role,
                        "parts": [{"text": msg.content}]
                    })

            chat = model.start_chat(history=gemini_history)
            response = await asyncio.to_thread(chat.send_message, message)
            
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Response error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi Gemini API: {str(e)}")

# Khởi tạo chatbot instance
chatbot = GeminiChatbot()

@app.get("/")
async def root():
    """Root endpoint - hướng dẫn sử dụng"""
    return {
        "message": "🤖 Gemini 2.0 Flash Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "stream_chat": "POST /api/chat/stream",
            "regular_chat": "POST /api/chat", 
            "health": "GET /api/health",
            "docs": "GET /docs"
        },
        "frontend": "/static/index.html",
        "status": "✅ API đang hoạt động"
    }

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Stream chat endpoint - trả về response từng chunk
    
    Request body:
    - message: Tin nhắn người dùng
    - history: Lịch sử chat (optional)
    - model: Model name (optional, default: gemini-2.0-flash-exp)
    - temperature: Độ sáng tạo 0-1 (optional, default: 0.7)
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="❌ GEMINI_API_KEY chưa được cấu hình. Vui lòng set environment variable."
        )
    
    logger.info(f"🚀 Starting stream for message: {request.message[:50]}...")
    
    return StreamingResponse(
        chatbot.stream_response(request.message, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Tắt buffering cho nginx
        }
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Regular chat endpoint - trả về response một lần
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="❌ GEMINI_API_KEY chưa được cấu hình"
        )
    
    logger.info(f"💬 Processing chat: {request.message[:50]}...")
    
    response = await chatbot.get_response(request.message, request.history)
    
    return ChatResponse(
        response=response,
        model=request.model or "gemini-2.0-flash-exp",
        timestamp=datetime.now()
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if GEMINI_API_KEY else "warning",
        "timestamp": datetime.now().isoformat(),
        "model": "gemini-2.0-flash-exp",
        "api_configured": bool(GEMINI_API_KEY),
        "message": "✅ API đang hoạt động bình thường" if GEMINI_API_KEY else "⚠️ Chưa cấu hình GEMINI_API_KEY"
    }

@app.get("/api/models")
async def get_available_models():
    """Danh sách models có sẵn"""
    return {
        "available_models": [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "current_model": chatbot.model_name,
        "generation_config": chatbot.generation_config
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "message": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"❌ Unhandled error: {str(exc)}")
    return {
        "error": True,
        "status_code": 500,
        "message": "Đã xảy ra lỗi không mong muốn",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting Gemini Chatbot Server...")
    print("📋 Thông tin:")
    print(f"   • Model: {chatbot.model_name}")
    print(f"   • API Key: {'✅ Đã cấu hình' if GEMINI_API_KEY else '❌ Chưa cấu hình'}")
    print(f"   • Server: http://localhost:8000")
    print(f"   • API Docs: http://localhost:8000/docs")
    print(f"   • Frontend: http://localhost:8000/static/index.html")
    print("\n💡 Cách set API key: export GEMINI_API_KEY='your-api-key'\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )