import json
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.generativeai import GenerativeModel, configure


# === Gemini LLM Wrapper ===
class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash", api_key=None):
        configure(api_key=api_key or os.getenv("AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"))
        self.model = GenerativeModel(model)

    def invoke(self, prompt: str) -> str:
        print("📝 Prompt Sent:\n", prompt)
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("✗ Gemini error:", e)
            return "Error: Cannot generate response."


# === Dataset Answerer ===
class DatasetAnswerer:
    def __init__(self, dataset_path="filtered_output.json", llm=None, min_similarity=0.25):
        self.llm = llm
        self.min_similarity = min_similarity

        # โหลด dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # เตรียมเวกเตอร์
        self.questions = [item["question"] for item in self.data]
        self.vectorizer = TfidfVectorizer().fit(self.questions)
        self.embeddings = self.vectorizer.transform(self.questions)

    def find_similar_context(self, query, top_n=3) -> List[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings)[0]

        # ดึงรายการที่คล้ายที่สุดและเกิน threshold
        top_indices = [i for i in scores.argsort()[::-1] if scores[i] >= self.min_similarity][:top_n]
        return [self.data[i] for i in top_indices]

    def answer(self, user_question: str) -> str:
        context_items = self.find_similar_context(user_question)

        if not context_items:
            return "Not found in dataset."

        # สร้างข้อความ context
        context_text = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in context_items]
        )

        # สร้าง prompt
        prompt = f"""
You are an AI assistant that answers questions based strictly on the provided dataset.

Here is the dataset context:
{context_text}

Now answer the following question in Thai:

{user_question}

If the answer is not in the dataset, say "Not found in dataset."
"""

        return self.llm.invoke(prompt)


# === Example Usage ===
if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"  # 🔐 ใส่ API Key ของคุณ

    # สร้าง LLM และ Answerer
    llm = GeminiLLM(api_key=GEMINI_API_KEY)
    answerer = DatasetAnswerer(dataset_path="filtered_output.json", llm=llm)

    # 🔍 ลองถามคำถาม
    questions = [
        "ตามประกาศ ธปท. ที่ สนส.15/2563 การแต่งตั้งตัวแทนทางการเงิน (Banking Agent) เพื่อให้บริการรับฝากเงิน มีหลักเกณฑ์กำกับดูแลที่สำคัญอย่างไรบ้าง?",
        "ตามประกาศ ธปท. สนส.15/2563 ตัวแทนรับฝากเงินสามารถให้บริการอะไรได้บ้างบอกหน่อยจ้า?"
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        print("✅ Answer:", answerer.answer(q))
        print("-" * 60)
