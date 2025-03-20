from embedding import get_embedding  # Import hàm tạo embedding
from deepseek_api import call_deepseek_ai  # Gọi AI từ deepseek_api.py

# Hàm xử lý câu hỏi của người dùng
def process_user_question(question):
    """
    Nhận câu hỏi từ người dùng, tạo embedding và gọi AI để lấy câu trả lời.

    Parameters:
        question (str): Câu hỏi của người dùng.

    Returns:
        dict: Kết quả gồm câu hỏi, embedding và câu trả lời từ AI.
    """
    try:
        question_embedding = get_embedding(question)  # Chuyển câu hỏi thành vector
        response = call_deepseek_ai(question)  # Gọi API DeepSeek để lấy câu trả lời

        return {
            "question": question,
            "embedding": question_embedding,
            "response": response
        }
    except Exception as e:
        return {"error": str(e)}
