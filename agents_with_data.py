import json
import sqlite3
import os
import re
import pandas as pd
from typing import List, Dict
from google.generativeai import GenerativeModel, configure


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
                answer_length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filtered_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_id INTEGER,
                content TEXT,
                quality_score REAL,
                filter_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_id) REFERENCES raw_data(id)
            )
        ''')

        self.conn.commit()
        print("✅ Database setup complete")

    def agent_1_from_csv(self, file_path: str, max_rows: int = 5, samples_per_row: int = 2) -> List[Dict]:
        print(f"📄 Agent 1: Creating Q&A from CSV file: {file_path}")
        df = pd.read_csv(file_path)
        df = df.head(max_rows)  # ✅ อ่านแค่ N แถวแรก

        created_samples = []
        cursor = self.conn.cursor()

        for i, row in df.iterrows():
            row_text = str(row.to_dict())

            for j in range(samples_per_row):
                prompt = f"""
                สร้างคำถาม-คำตอบ (ภาษาไทย) จากเนื้อหาต่อไปนี้:

                {row_text}

                - คำถามต้องตรงกับเนื้อหา
                - คำตอบไม่เกิน 150 คำ
                - ต้องไม่ซ้ำกับคำถามก่อนหน้า
                - รูปแบบ JSON เท่านั้น:
                {{
                "question": "…?",
                "answer": "…"
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
                        print(f"  ⚠️ Row {i+1}.{j+1} skipped: Question missing '?'")
                        continue

                    word_count = len((question + " " + answer).split())
                    cursor.execute(
                        "INSERT INTO raw_data (content, data_type, word_count, question_length, answer_length) VALUES (?, ?, ?, ?, ?)",
                        (json.dumps(sample, ensure_ascii=False), "from_csv", word_count, len(question), len(answer))
                    )
                    created_samples.append(sample)
                    print(f"  ✓ Created Q&A from row {i+1}, version {j+1}")

                except Exception as e:
                    print(f"  ✗ Row {i+1}, version {j+1} error: {e}")

        self.conn.commit()
        return created_samples



    # === agent_2_quality_filter, agent_3_main_trainer, export_data, get_statistics ตามที่คุณมี ===
    # (ขอยกเว้นไว้ตรงนี้เพื่อไม่ให้ข้อความยาวเกิน)    
    def agent_2_quality_filter(self, min_quality_score: float = 7.0) -> List[Dict]:
        print(f"🔍 Agent 2: Filtering with min score {min_quality_score}...")
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
        print(f"✅ Agent 2: {len(filtered)} samples passed filter")
        return filtered

    def agent_3_main_trainer(self) -> Dict:
        print("🎯 Agent 3: Simulating training...")
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
        and give a answer in thai
        """
        analysis = self.llm.invoke(prompt)

        return {
            "status": "success",
            "count": len(samples),
            "average_score": avg,
            "summary": analysis,
            "examples": samples[:2]
        }

    def export_data(self, filename="training_data.json"):
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

        print(f"📁 Exported {len(export)} samples to {filename}")

    def get_statistics(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(word_count), AVG(question_length), AVG(answer_length) FROM raw_data")
        count, avg_words, avg_qlen, avg_alen = cursor.fetchone()
        print("📊 Stats:")
        print(f"- Total Samples: {count}")
        print(f"- Avg Words: {avg_words:.2f}")
        print(f"- Avg Question Length: {avg_qlen:.2f}")
        print(f"- Avg Answer Length: {avg_alen:.2f}")


    def run_pipeline_from_excel(self, excel_path: str, sheet_name: str = None, min_quality: float = 7.5):
        print(f"\n🚀 Starting Excel-based Multi-Agent Pipeline: {excel_path}")
        print("="*50)

        print("STEP 1: IMPORT + Q&A CREATION")
        raw = self.agent_1_from_excel(excel_path, sheet_name)
        if not raw:
            print("❌ No data generated from Excel")
            return

        print("\nSTEP 2: QUALITY FILTER")
        filtered = self.agent_2_quality_filter(min_quality)
        if not filtered:
            print("❌ No samples passed quality check")
            return

        print("\nSTEP 3: SIMULATE TRAINING")
        result = self.agent_3_main_trainer()
        if result["status"] == "success":
            print("\n📊 Summary:")
            print(f"✓ {result['count']} samples used")
            print(f"✓ Avg score: {result['average_score']:.2f}")
            print(result['summary'])

        self.export_data("filtered_output.json")
        self.get_statistics()


# =========================
# ✅ Usage
# =========================

    def run_full_pipeline_from_csv(self, csv_path: str, min_quality: float = 7.5):
        print(f"\n🚀 Starting CSV-based Multi-Agent Pipeline: {csv_path}")
        print("="*50)

        print("STEP 1: DATA CREATION")
        raw = self.agent_1_from_csv(csv_path)
        if not raw:
            print("❌ No data generated")
            return

        print("\nSTEP 2: QUALITY FILTER")
        filtered = self.agent_2_quality_filter(min_quality)
        if not filtered:
            print("❌ No samples passed filter")
            return

        print("\nSTEP 3: SIMULATE TRAINING")
        result = self.agent_3_main_trainer()
        if result["status"] == "success":
            print("\n📊 Summary:")
            print(f"✓ {result['count']} samples used")
            print(f"✓ Avg score: {result['average_score']:.2f}")
            print(result['summary'])

        self.export_data("filtered_output.json")
        self.get_statistics()

if __name__ == "__main__":
    system = MultiAgentSystem(
        model_name="gemini-2.5-flash",
        api_key="AIzaSyDjGO4GS5GPbO0-fqX_8qr6M0qhY6BFys0"
    )
    system.run_full_pipeline_from_csv("/Users/user/Documents/2024intern/DF8ADB5EC86B354685F1B24EA0AB4BE36EDF5DB3_HC_Main_Data3 (1).csv", min_quality=7.5)

