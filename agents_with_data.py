import json
import sqlite3
import os
import re
import pandas as pd
from typing import List, Dict
from google.generativeai import GenerativeModel, configure
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash", api_key=None):
        configure(api_key=api_key or os.getenv("AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"))
        self.model = GenerativeModel(model)

    def invoke(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("✗ Gemini error:", e)
            return ""


class MultiAgentSystem:
    def __init__(self, model_name="gemini-2.5-flash", api_key=None):
        self.model_name = model_name
        self.llm = GeminiLLM(model=model_name, api_key=api_key)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists("agent_data.db"):
            os.remove("agent_data.db")
            print("🗑️ Deleted old database file")

        self.setup_database()

    def setup_database(self):
        self.conn = sqlite3.connect("agent_data.db")
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                data_type TEXT,
                word_count INTEGER,
                question_length INTEGER,
                answer_length INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filtered_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_id INTEGER,
                content TEXT,
                quality_score REAL,
                filter_reason TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()
        print("✅ Database setup complete (QnA + RAG)")

    def agent_1_from_csv(self, file_path: str, max_rows: int = 5):
        print(f"📄 Agent 1: Generating Q&A from CSV: {file_path}")
        df = pd.read_csv(file_path)
        df = df.head(max_rows)
        cursor = self.conn.cursor()
        created_samples = []

        for i, row in df.iterrows():
            row_text = str(row.to_dict())

            for j in range(2):
                prompt = f"""
                สร้างคำถาม-คำตอบ (ภาษาไทย) จากเนื้อหาต่อไปนี้:

                {row_text}

                - คำถามต้องตรงกับเนื้อหา
                - คำตอบไม่เกิน 150 คำ
                - ต้องไม่ซ้ำกับคำถามก่อนหน้า
                - รูปแบบ JSON เท่านั้น:
                {{
                "question": "...?",
                "answer": "..."
                }}
                """
                try:
                    response = self.llm.invoke(prompt)
                    start = response.find('{')
                    end = response.rfind('}') + 1
                    json_str = response[start:end]
                    sample = json.loads(json_str)
                    question = sample.get("question", "")
                    answer = sample.get("answer", "")

                    if not question.endswith("?"):
                        continue

                    word_count = len((question + " " + answer).split())
                    cursor.execute(
                        "INSERT INTO raw_data (content, data_type, word_count, question_length, answer_length) VALUES (?, ?, ?, ?, ?)",
                        (json.dumps(sample, ensure_ascii=False), "from_csv", word_count, len(question), len(answer))
                    )
                    created_samples.append(sample)

                except Exception as e:
                    print(f"  ✗ Row {i+1}, version {j+1} error: {e}")

        self.conn.commit()
        print("✅ Q&A Generated")
        return created_samples

    def agent_2_quality_filter(self, min_quality_score: float = 7.0) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content FROM raw_data")
        raw_samples = cursor.fetchall()

        filtered = []
        for sample_id, content_str in raw_samples:
            content = json.loads(content_str)
            q = content['question']
            a = content['answer']

            eval_prompt = f"""
            Evaluate the following question-answer pair:

            Question: {q}
            Answer: {a}

            Provide these scores (1–10 only):
            - Accuracy:
            - Clarity:
            - Completeness:
            - Grammar:
            Then give:

            Overall Score: (1–10 only)
            Reason: A brief reason in 1–2 sentences

            Output format:
            Accuracy: X
            Clarity: X
            Completeness: X
            Grammar: X
            Overall Score: X
            Reason: Y
            """
            try:
                result = self.llm.invoke(eval_prompt)
                match = re.search(r'Overall Score[:\s]*([\d.]+)', result)
                score_raw = float(match.group(1)) if match else 5.0
                score = min(score_raw, 10.0)
                reason = result.split("Reason:")[-1].strip() if "Reason:" in result else "N/A"

                if score >= min_quality_score:
                    cursor.execute(
                        "INSERT INTO filtered_data (raw_id, content, quality_score, filter_reason) VALUES (?, ?, ?, ?)",
                        (sample_id, content_str, score, reason)
                    )
                    filtered.append(content)

            except Exception as e:
                print(f"  ✗ Filter error: {e}")

        self.conn.commit()
        return filtered

    def agent_3_main_trainer(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT content, quality_score FROM filtered_data")
        data = cursor.fetchall()

        if not data:
            return {"status": "error", "message": "No filtered data available"}

        samples = []
        total_score = 0
        for content_str, score in data:
            content = json.loads(content_str)
            samples.append(content)
            total_score += score

        avg = total_score / len(samples)
        prompt = f"""
        Analyze training data of {len(samples)} pairs.
        Avg score: {avg:.2f}
        Example questions: {[s['question'][:40] for s in samples[:3]]}

        Give summary with:
        - Strengths
        - Weaknesses
        - Next steps
        (ตอบเป็นภาษาไทย)
        """
        analysis = self.llm.invoke(prompt)

        return {
            "status": "success",
            "count": len(samples),
            "average_score": avg,
            "summary": analysis,
            "examples": samples[:2]
        }

    def agent_rag_import(self, file_path: str, max_rows: int = 5):
        print(f"📄 Agent RAG: Importing and embedding from CSV: {file_path}")
        df = pd.read_csv(file_path)
        df = df.head(max_rows)
        cursor = self.conn.cursor()

        for i, row in df.iterrows():
            row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
            if len(row_text.strip()) < 10:
                continue
            embedding = self.embedder.encode(row_text).astype(np.float32).tobytes()
            cursor.execute("INSERT INTO context_data (content, embedding) VALUES (?, ?)", (row_text, embedding))

        self.conn.commit()
        print("✅ RAG Context Imported")

    def rag_answer(self, user_question: str, top_k: int = 3) -> str:
        query_vec = self.embedder.encode([user_question]).astype(np.float32)
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, embedding FROM context_data")
        rows = cursor.fetchall()

        print(f"🔎 มี context ในฐานข้อมูล: {len(rows)} รายการ")
        if not rows:
            return "❌ ไม่มีข้อมูล context ในฐานข้อมูล"

        contexts = [(content, np.frombuffer(embedding, dtype=np.float32)) for _, content, embedding in rows]
        matrix = np.vstack([vec for _, vec in contexts])

        print(f"🔎 embedding matrix shape: {matrix.shape}")
        sims = cosine_similarity(query_vec, matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]

        print(f"🔎 Top similarity scores: {[sims[i] for i in top_indices]}")

        context_text = "\n".join([f"- {contexts[i][0]}" for i in top_indices])

        prompt = f"""
        คุณคือตัวช่วย AI ที่ตอบคำถามจากข้อมูลที่ให้เท่านั้น

        ข้อมูล:
        {context_text}

        คำถาม: {user_question}

        หากไม่มีข้อมูลให้ตอบว่า "ไม่พบข้อมูลใน dataset"
        """
        return self.llm.invoke(prompt)


    def export_data(self, filename="filtered_output.json"):
        cursor = self.conn.cursor()
        cursor.execute("SELECT content, quality_score FROM filtered_data")
        data = cursor.fetchall()

        export = []
        for content_str, score in data:
            content = json.loads(content_str)
            content['quality_score'] = score
            export.append(content)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

    def get_statistics(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(word_count), AVG(question_length), AVG(answer_length) FROM raw_data")
        count, avg_words, avg_qlen, avg_alen = cursor.fetchone()
        print("📊 Stats:")
        print(f"- Total Samples: {count}")
        print(f"- Avg Words: {avg_words:.2f}")
        print(f"- Avg Question Length: {avg_qlen:.2f}")
        print(f"- Avg Answer Length: {avg_alen:.2f}")

    def run_full_pipeline_from_csv(self, csv_path: str, min_quality: float = 7.5, mode: str = "qna"):
        print(f"\n🚀 Starting Multi-Agent Pipeline in mode: {mode.upper()}\n{'='*50}")

        if mode == "qna":
            raw = self.agent_1_from_csv(csv_path)
            if not raw:
                print("❌ No data generated")
                return

            filtered = self.agent_2_quality_filter(min_quality)
            if not filtered:
                print("❌ No samples passed filter")
                return

            result = self.agent_3_main_trainer()
            if result["status"] == "success":
                print("\n📊 Summary:")
                print(f"✓ {result['count']} samples used")
                print(f"✓ Avg score: {result['average_score']:.2f}")
                print(result['summary'])

            self.export_data("filtered_output.json")
            self.get_statistics()

        elif mode == "rag":
            self.agent_rag_import(csv_path)
            print("✅ RAG pipeline complete")
        else:
            print("❌ Invalid mode. Use 'qna' or 'rag'")


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"
    system = MultiAgentSystem(model_name="gemini-2.5-flash", api_key=api_key)

    csv_file = "/Users/user/Documents/2024intern/DF8ADB5EC86B354685F1B24EA0AB4BE36EDF5DB3_HC_Main_Data3 (1).csv"

    # Mode QnA
    system.run_full_pipeline_from_csv(csv_file, min_quality=7.5, mode="qna")

    # Mode RAG
    system.run_full_pipeline_from_csv(csv_file, mode="rag")

    # Example RAG Query
    while True:
        user_question = input("\n💬 พิมพ์คำถามของคุณ (หรือพิมพ์ 'exit' เพื่อออก): ")
        if user_question.strip().lower() in ["exit", "quit"]:
            print("👋 จบการทำงานของระบบ RAG แล้ว")
            break

        answer = system.rag_answer(user_question)
        print("\n🔍 คำตอบจาก RAG:")
        print(answer)