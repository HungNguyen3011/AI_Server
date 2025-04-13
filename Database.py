import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer
from deepseek_api import call_deepseek_ai  # Hàm gọi API DeepSeek, đã tách riêng ở file deepseek_api.py

# Tạo instance FastAPI
app = FastAPI()

# Cấu hình CORS cho phép tất cả nguồn truy cập (để dùng trên website như Wix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo ChromaDB, sử dụng collection "library_collection" để lưu trữ library hợp nhất
chroma_client = chromadb.PersistentClient(path="./chroma_db")
library_collection = chroma_client.get_or_create_collection("library_collection")

# Khởi tạo mô hình embedding (sử dụng chung cho toàn hệ thống)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()

# --- MODULE: Document Loader (Hợp nhất dữ liệu mới vào library) ---
def merge_document_to_library(new_content: str):
    """
    Hợp nhất nội dung mới (new_content) vào một tài liệu duy nhất (library).
    Nếu library đã tồn tại, nội dung mới được nối vào sau nội dung hiện có.
    Sau đó tạo embedding mới và cập nhật vào ChromaDB.
    """
    library_id = "library"  # ID duy nhất cho kho dữ liệu tổng hợp
    try:
        current = library_collection.get(ids=[library_id])
    except Exception as e:
        current = None

    if current and current.get("documents") and len(current["documents"]) > 0:
        current_text = current["documents"][0]
        merged_text = current_text + "\n" + new_content
    else:
        merged_text = new_content

    new_embedding = get_embedding(merged_text)

    # Xóa tài liệu cũ (nếu có) và thêm lại với nội dung hợp nhất
    try:
        library_collection.delete(ids=[library_id])
    except Exception as e:
        pass

    library_collection.add(
        ids=[library_id],
        documents=[merged_text],
        embeddings=[new_embedding],
        metadatas=[{"source": "merged_library"}]
    )
    return {"message": "Library updated successfully.", "merged_length": len(merged_text)}

# --- END MODULE Document Loader ---

# --- MODULE: Chat History (nếu cần, demo rất đơn giản) ---
# Nếu bạn muốn lưu lịch sử chat vào file hoặc một cơ sở dữ liệu, bạn có thể tích hợp sau.
# Demo dưới đây chỉ trả về một biến toàn cục (không lưu cục bộ).
chat_history = []

def add_chat_message(sender: str, text: str):
    chat_history.append({"sender": sender, "text": text})

def clear_chat_history():
    global chat_history
    chat_history = []

# --- END MODULE Chat History ---

# --- Pydantic Models ---
class Question(BaseModel):
    question: str

class DocumentData(BaseModel):
    content: str

# --- END Models ---

# Endpoint cơ bản
@app.get("/")
def read_root():
    return {"message": "API is running."}

# Endpoint trả lời câu hỏi
@app.post("/ask")
def ask_question_endpoint(data: Question):
    user_question = data.question
    # (Có thể thêm logic tìm kiếm tài liệu từ library ở đây nếu cần)
    ai_response = call_deepseek_ai(user_question)
    # Lưu trữ chat history nếu muốn
    add_chat_message("user", user_question)
    add_chat_message("ai", ai_response)
    return {"response": ai_response}

# Endpoint cập nhật Library: hợp nhất nội dung mới vào kho dữ liệu
@app.post("/documents/update_library")
def update_library_endpoint(doc: DocumentData):
    result = merge_document_to_library(doc.content)
    return result

# Endpoint hiển thị trạng thái AI: lấy thông tin từ library (ví dụ, phần đầu của nội dung)
@app.get("/status")
def get_status():
    try:
        result = library_collection.get(ids=["library"])
        if result and result.get("documents") and len(result["documents"]) > 0:
            info = result["documents"][0][:500]  # Ví dụ trả về 500 ký tự đầu
        else:
            info = "No library data available. AI uses its own internal info."
        return {"status": "AI is ready", "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint quản lý chat history
@app.get("/chat_history")
def get_chat_history():
    return {"chat_history": chat_history}

@app.delete("/chat_history")
def delete_chat_history():
    clear_chat_history()
    return {"message": "Chat history cleared."}
