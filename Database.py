import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding import get_embedding
from deepseek_api import call_deepseek_ai  # Gọi AI từ deepseek_api.py
from ai_service import process_user_question  # Gọi hàm xử lý câu hỏi từ ai_service.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn truy cập (có thể đổi thành domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

class Question(BaseModel):
    question: str

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db/")

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get("/embedding/")
def embedding_endpoint(text: str):
    return {"embedding": get_embedding(text)}

@app.post("/ask/")
async def ask_question(data: Question):
    try:
        result = process_user_question(data.question)  # Xử lý câu hỏi
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
