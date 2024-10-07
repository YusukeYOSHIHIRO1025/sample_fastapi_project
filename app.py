from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
from typing import List

# 環境変数からOpenAI APIキーを取得
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 文書データベースとFaissインデックス
documents = []
index = faiss.IndexFlatL2(1536)  # OpenAIの埋め込みは1536次元

class ChatRequest(BaseModel):
    question: str

class DataProcessingRequest(BaseModel):
    data: dict

class Document(BaseModel):
    content: str

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

@app.post("/add-document")
async def add_document(document: Document):
    global documents, index
    embedding = get_embedding(document.content)
    documents.append(document.content)
    index.add(np.array([embedding]))
    return {"message": "Document added successfully"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    question = request.question

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        # 質問の埋め込みを取得
        question_embedding = get_embedding(question)
        
        # Faissで最も関連性の高い文書を検索
        k = 1  # 最も関連性の高い1つの文書を取得
        D, I = index.search(np.array([question_embedding]), k)
        
        relevant_doc = documents[I[0][0]] if len(documents) > 0 else ""

        # OpenAIのChat APIを使用して質疑応答
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the question."},
                {"role": "user", "content": f"Context: {relevant_doc}\n\nQuestion: {question}"}
            ],
            max_tokens=150
        )
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