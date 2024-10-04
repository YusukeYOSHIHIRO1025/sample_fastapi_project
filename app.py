from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from fastapi.middleware.cors import CORSMiddleware  # CORSミドルウェアをインポート

# 環境変数からOpenAI APIキーを取得
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて特定のオリジンを許可できます
    allow_credentials=True,
    allow_methods=["*"],  # すべてのメソッド（POST, GET など）を許可
    allow_headers=["*"],
)

# リクエストボディのモデルを定義
class ChatRequest(BaseModel):
    question: str

class DataProcessingRequest(BaseModel):
    data: dict

@app.post("/api/chat")
async def chat(request: ChatRequest):
    question = request.question

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # OpenAIのChat APIを使用して質疑応答
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        # ChatGPTライクな応答を返す
        answer = response['choices'][0]['message']['content'].strip()
        return {"answer": answer}
    except openai.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/process-data")
async def process_data(request: DataProcessingRequest):
    try:
        processed_data = {key: value.upper() for key, value in request.data.items()}
        return {"processed_data": processed_data}
    except Exception as e:
        print(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
