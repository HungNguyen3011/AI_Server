import os
from dotenv import load_dotenv

# Load file .env
load_dotenv()

# Lấy API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Kiểm tra API Key có tồn tại không
if not DEEPSEEK_API_KEY:
    raise ValueError("API Key không được tìm thấy! Hãy kiểm tra biến môi trường.")