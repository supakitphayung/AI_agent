import json
import os
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.generativeai import GenerativeModel, configure


# === Gemini LLM Wrapper ===
class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash", api_key=None):
        configure(api_key=api_key or os.getenv("AIzaSyA_xKBxU1sV3hk2alE0PWofxI1l3PwiTJ0"))
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
    def __init__(self, dataset_path="filtered_output.json", llm=None):
        self.llm = llm
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.questions = [item["question"] for item in self.data]
        self.vectorizer = TfidfVectorizer().fit(self.questions)
        self.embeddings = self.vectorizer.transform(self.questions)

    def find_similar_context(self, query, top_n=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = scores.argsort()[::-1][:top_n]
        return [self.data[i] for i in top_idx]

    def answer(self, user_question: str) -> str:
        context_items = self.find_similar_context(user_question)
        context_text = "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in context_items]
        )

        prompt = f"""
You are an AI that can only answer based on the provided datase

Now answer this question:
{user_question}

If the answer is not in the dataset, say "Not found in dataset."
"""
        return self.llm.invoke(prompt)


# === Example Usage ===
if __name__ == "__main__":
    # 🛠 Replace with your actual Gemini API Key
    GEMINI_API_KEY = "AIzaSyA_xKBxU1sV3hk2alE0PWofxI1l3PwiTJ0"

    # ✅ Create LLM instance
    llm = GeminiLLM(api_key=GEMINI_API_KEY)

    # ✅ Create dataset answerer
    answerer = DatasetAnswerer(dataset_path="filtered_output.json", llm=llm)

    # 🧪 Ask new questions
    questions = [
        "ตามประกาศ ธปท. ที่ สนส.15/2563 การแต่งตั้งตัวแทนทางการเงิน (Banking Agent) เพื่อให้บริการรับฝากเงิน มีหลักเกณฑ์กำกับดูแลที่สำคัญอย่างไรบ้าง?"
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        print("✅ Answer:", answerer.answer(q))
        print("-" * 60)
