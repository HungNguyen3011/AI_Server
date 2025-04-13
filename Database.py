import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding import get_embedding
from deepseek_api import call_deepseek_ai  # Gọi AI từ deepseek_api.py
from ai_service import process_user_question  # Gọi hàm xử lý câu hỏi từ ai_service.py
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# Import module đọc file PDF
from document_loader import load_pdf_to_chroma

# Khởi tạo ứng dụng FastAPI và cấu hình CORS (nếu chưa có)
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

class ChatQuestion(BaseModel):
    question: str

@app.post("/ask")
def ask_question_endpoint(data: ChatQuestion):
    user_question = data.question
    # Lưu câu hỏi vào chat history với dấu thời gian
    chat_history.append({"sender": "user", "text": user_question, "timestamp": datetime.utcnow().isoformat()})
    
    # Gọi AI để lấy câu trả lời (có thể kết hợp với tìm kiếm dữ liệu tài liệu qua ChromaDB nếu cần mở rộng)
    ai_response = call_deepseek_ai(user_question)
    
    # Lưu phản hồi của AI vào chat history
    chat_history.append({"sender": "ai", "text": ai_response, "timestamp": datetime.utcnow().isoformat()})
    
    return {"response": ai_response}

@app.post("/load_documents")
def load_documents():
    # Ví dụ: file PDF đã được upload vào thư mục "documents"
    pdf_file = "DATA.pdf"  # Điều chỉnh đường dẫn nếu cần
    result = load_pdf_to_chroma(pdf_file)
    return result

chat_history = []

@app.get("/chat_history")
def get_chat_history():
    return {"chat_history": chat_history}

@app.delete("/chat_history")
def clear_chat_history():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared."}

@app.get("/status")
def get_status():
    # Kết nối ChromaDB và tìm kiếm thông tin về AI từ các tài liệu đã load
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("documents")
    # Demo: tìm kiếm với từ khóa "Thông tin AI" (bạn cần đảm bảo tài liệu của bạn có chứa thông tin này)
    results = collection.query(query_texts=["Thông tin AI"], n_results=1)
    if results and results.get("documents") and len(results["documents"][0]) > 0:
        info = results["documents"][0][0]
    else:
        info = "Không có thông tin tài liệu. AI trả lời theo cách riêng."
    return {"status": "AI is ready", "info": info}


