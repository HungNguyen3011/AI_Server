import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding import get_embedding
from deepseek_api import call_deepseek_ai  # Gọi AI từ deepseek_api.py
from ai_service import process_user_question  # Gọi hàm xử lý câu hỏi từ ai_service.py
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Thêm CORS middleware để cho phép truy cập từ các domain khác (nếu cần)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn truy cập (có thể đổi thành domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db/")

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/ask/")
async def ask_question_endpoint(data: Question):
    try:
        # Tạo embedding từ câu hỏi
        question_embedding = get_embedding(data.question)
        
        # Gọi AI để lấy câu trả lời (chuyển sang hàm call_deepseek_ai nếu đã tích hợp)
        ai_response = call_deepseek_ai(data.question)
        
        return {
            "question": data.question,
            "embedding": question_embedding,
            "response": ai_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))