import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from agents_with_data import MultiAgentSystem  # import คลาสจากไฟล์หลักของคุณ

def evaluate_rag_accuracy_from_json(system, file_path: str, top_k: int = 3, threshold: float = 0.6):
    with open(file_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    total = len(test_cases)
    correct = 0

    for i, item in enumerate(test_cases):
        question = item["question"]
        expected = item["answer"]
        predicted = system.rag_answer(question, top_k=top_k)

        vec_expected = embedder.encode([expected])[0].reshape(1, -1)
        vec_predicted = embedder.encode([predicted])[0].reshape(1, -1)
        sim = cosine_similarity(vec_expected, vec_predicted)[0][0]

        print(f"\n{i+1}) Q: {question}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Similarity: {sim:.2f}")

        if sim >= threshold:
            correct += 1

    accuracy = correct / total * 100
    print(f"\n🎯 Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    # กำหนด API Key และ CSV Path ให้เหมือนใน main_system.py
    api_key = "AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"  # หรือ os.getenv(...) แบบในไฟล์หลัก

    system = MultiAgentSystem(
        model_name="gemini-2.5-flash",
        api_key=api_key
    )

    csv_file = "/Users/user/Documents/2024intern/DF8ADB5EC86B354685F1B24EA0AB4BE36EDF5DB3_HC_Main_Data3 (1).csv"

    # โหลดข้อมูลจาก CSV เพื่อสร้างฐานข้อมูล context
    system.agent_1_from_csv(csv_file)

    # เรียกฟังก์ชันเทสต์ accuracy
    test_file = "filtered_output.json"
    evaluate_rag_accuracy_from_json(system, test_file)
