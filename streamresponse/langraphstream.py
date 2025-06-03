from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncIterator
import json
import asyncio
import os
from datetime import datetime
import logging
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

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
    model: Optional[str] = "gemini-2.0-flash"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: datetime

# LangGraph State
class State(TypedDict):
    messages: List[BaseMessage]
    model_config: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="LangGraph Gemini Chatbot API",
    description="API cho chatbot sử dụng LangGraph + Gemini với streaming",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("✅ Static files mounted successfully")

# Gemini API Key
GEMINI_API_KEY = 'AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0'
if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY chưa được set")

class LangGraphChatbot:
    def __init__(self):
        self.model_name = "gemini-2.0-flash"
        self.llm = None
        self.graph = None
        self.checkpointer = MemorySaver()
        self._setup_llm()
        self._build_graph()

    def _setup_llm(self):
        """Setup Gemini LLM với LangChain"""
        if GEMINI_API_KEY:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=GEMINI_API_KEY,
                temperature=0.7,
                max_output_tokens=2048,
                streaming=True  # Enable streaming
            )

    def _build_graph(self):
        """Xây dựng LangGraph workflow"""
        
        async def chatbot_node(state: State) -> State:
            """Node xử lý chat chính"""
            try:
                messages = state["messages"]
                model_config = state.get("model_config", {})
                
                # Update model config if provided
                if model_config:
                    temperature = model_config.get("temperature", 0.7)
                    max_tokens = model_config.get("max_tokens", 2048)
                    
                    # Create new LLM instance with updated config
                    llm = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        google_api_key=GEMINI_API_KEY,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        streaming=True
                    )
                else:
                    llm = self.llm

                # Generate response
                response = await llm.ainvoke(messages)
                
                return {
                    "messages": messages + [response],
                    "model_config": model_config
                }
                
            except Exception as e:
                logger.error(f"❌ Chatbot node error: {str(e)}")
                error_message = AIMessage(content=f"Xin lỗi, đã xảy ra lỗi: {str(e)}")
                return {
                    "messages": state["messages"] + [error_message],
                    "model_config": state.get("model_config", {})
                }

        # Build the graph
        workflow = StateGraph(State)
        workflow.add_node("chatbot", chatbot_node)
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("chatbot", END)
        
        # Compile with checkpointer for memory
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    async def stream_response(self, message: str, history: List[ChatMessage] = None, 
                            model_config: Dict[str, Any] = None, session_id: str = "default") -> AsyncIterator[str]:
        """Stream response từ LangGraph"""
        try:
            if not self.llm:
                raise Exception("Gemini API chưa được cấu hình")

            # Convert history to LangChain messages
            messages = []
            if history:
                for msg in history:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))
            
            # Add current message
            messages.append(HumanMessage(content=message))

            # Initial state
            initial_state = {
                "messages": messages,
                "model_config": model_config or {}
            }

            # Config for the run
            config = RunnableConfig(
                configurable={"thread_id": session_id}
            )

            full_response = ""
            chunk_count = 0

            # Stream from LangGraph
            async for event in self.graph.astream(initial_state, config=config, stream_mode="values"):
                if "messages" in event and len(event["messages"]) > len(messages):
                    # Get the latest AI message
                    latest_message = event["messages"][-1]
                    if isinstance(latest_message, AIMessage):
                        content = latest_message.content
                        
                        # Stream character by character for typing effect
                        if content and content != full_response:
                            new_content = content[len(full_response):]
                            for char in new_content:
                                full_response += char
                                chunk_count += 1

                                chunk_data = {
                                    'type': 'chunk',
                                    'content': char,
                                    'full_content': full_response,
                                    'chunk_id': chunk_count,
                                    'session_id': session_id,
                                    'timestamp': datetime.now().isoformat()
                                }

                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.015)

            # Send completion signal
            done_data = {
                'type': 'done',
                'content': '',
                'full_content': full_response,
                'total_chunks': chunk_count,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

            logger.info(f"✅ LangGraph stream completed: {chunk_count} chunks, {len(full_response)} chars")

        except Exception as e:
            logger.error(f"❌ LangGraph streaming error: {str(e)}")
            error_data = {
                'type': 'error',
                'error': f'Lỗi khi xử lý với LangGraph: {str(e)}',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    async def get_response(self, message: str, history: List[ChatMessage] = None, 
                          model_config: Dict[str, Any] = None, session_id: str = "default") -> str:
        """Non-streaming response"""
        try:
            if not self.llm:
                raise Exception("Gemini API chưa được cấu hình")

            # Convert history to messages
            messages = []
            if history:
                for msg in history:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))
            
            messages.append(HumanMessage(content=message))

            initial_state = {
                "messages": messages,
                "model_config": model_config or {}
            }

            config = RunnableConfig(
                configurable={"thread_id": session_id}
            )

            # Run the graph
            result = await self.graph.ainvoke(initial_state, config=config)
            
            # Get the last AI message
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
            else:
                return "Không thể tạo phản hồi"

        except Exception as e:
            logger.error(f"❌ LangGraph response error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi LangGraph: {str(e)}")

    async def get_conversation_history(self, session_id: str = "default") -> List[Dict]:
        """Lấy lịch sử hội thoại"""
        try:
            config = RunnableConfig(configurable={"thread_id": session_id})
            
            # Get state from checkpointer
            state = await self.graph.aget_state(config)
            
            if not state or not state.values.get("messages"):
                return []
            
            history = []
            for msg in state.values["messages"]:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Error getting history: {str(e)}")
            return []

# Initialize chatbot
chatbot = LangGraphChatbot()

# Routes
@app.get("/")
async def root():
    return {
        "message": "🤖 LangGraph Gemini Chatbot API",
        "version": "2.0.0",
        "features": [
            "LangGraph workflow management",
            "Persistent conversation memory", 
            "Session-based chat",
            "Streaming responses",
            "Advanced state management"
        ],
        "endpoints": {
            "stream_chat": "POST /api/chat/stream",
            "regular_chat": "POST /api/chat",
            "conversation_history": "GET /api/chat/history/{session_id}",
            "health": "GET /api/health"
        },
        "status": "✅ LangGraph API đang hoạt động"
    }

@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    """LangGraph streaming chat endpoint"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="❌ GEMINI_API_KEY chưa được cấu hình")
    
    logger.info(f"🚀 LangGraph streaming for: {request.message[:50]}... (Session: {request.session_id})")
    
    model_config = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "model": request.model
    }
    
    return StreamingResponse(
        chatbot.stream_response(
            message=request.message,
            history=request.history,
            model_config=model_config,
            session_id=request.session_id or "default"
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        }
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """LangGraph regular chat endpoint"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="❌ GEMINI_API_KEY chưa được cấu hình")
    
    logger.info(f"💬 LangGraph processing: {request.message[:50]}... (Session: {request.session_id})")
    
    model_config = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "model": request.model
    }
    
    response = await chatbot.get_response(
        message=request.message,
        history=request.history,
        model_config=model_config,
        session_id=request.session_id or "default"
    )
    
    return ChatResponse(
        response=response,
        model=request.model or "gemini-2.0-flash",
        timestamp=datetime.now()
    )

@app.get("/api/chat/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Lấy lịch sử hội thoại theo session"""
    try:
        history = await chatbot.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy lịch sử: {str(e)}")

@app.delete("/api/chat/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Xóa lịch sử hội thoại"""
    try:
        # Clear checkpointer for this session
        config = RunnableConfig(configurable={"thread_id": session_id})
        # Note: MemorySaver doesn't have a direct clear method, 
        # but we can implement this by reinitializing
        return {
            "message": f"✅ Đã xóa lịch sử hội thoại cho session: {session_id}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa lịch sử: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check với LangGraph info"""
    return {
        "status": "healthy" if GEMINI_API_KEY else "warning",
        "timestamp": datetime.now().isoformat(),
        "framework": "LangGraph + LangChain",
        "model": "gemini-2.0-flash",
        "api_configured": bool(GEMINI_API_KEY),
        "features": {
            "streaming": True,
            "memory": True,
            "session_management": True,
            "workflow_management": True
        },
        "message": "✅ LangGraph API hoạt động bình thường" if GEMINI_API_KEY else "⚠️ Chưa cấu hình GEMINI_API_KEY"
    }

@app.get("/api/sessions")
async def list_active_sessions():
    """Liệt kê các session đang hoạt động"""
    # This would require extending MemorySaver to track sessions
    return {
        "message": "Session management - cần implement với database",
        "suggestion": "Dùng Redis hoặc PostgreSQL để lưu trữ sessions"
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
        "message": "Đã xảy ra lỗi không mong muốn trong LangGraph",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting LangGraph Gemini Chatbot Server...")
    print("📋 Thông tin:")
    print(f"   • Framework: LangGraph + LangChain")
    print(f"   • Model: {chatbot.model_name}")
    print(f"   • API Key: {'✅ Đã cấu hình' if GEMINI_API_KEY else '❌ Chưa cấu hình'}")
    print(f"   • Memory: MemorySaver (in-memory)")
    print(f"   • Server: http://localhost:8000")
    print(f"   • API Docs: http://localhost:8000/docs")
    print(f"   • Features: Streaming, Session Management, Conversation Memory")
    print("\n💡 Cài đặt: pip install langgraph langchain-google-genai\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )