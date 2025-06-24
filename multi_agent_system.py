import json
import sqlite3
import os
import re
from typing import List, Dict
from google.generativeai import GenerativeModel, configure


class GeminiLLM:
    def __init__(self, model="gemini-2.5-flash", api_key=None):
        configure(api_key=api_key or os.getenv("AIzaSyBzVdrsbkQL7qXmcSZNbVs8kN7jyVfqzF0"))
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

    def agent_1_data_creator(self, topic: str = "bank of thailand regulation", num_samples: int = 5) -> List[Dict]:
        print(f"🤖 Agent 1: Creating {num_samples} data samples about '{topic}'...")
        created_samples = []
        cursor = self.conn.cursor()

        for i in range(num_samples):
            prompt = f"""
            Forget previous context.

            Create a unique and different question-answer pair strictly about the topic: "{topic}".
            Each sample must focus on a different sub-topic ex. stats of premier league players or another league and don't give a rule of football.
            Avoid repeating questions and facts or the same field of questions. Ensure the answer is under 200 words and factually correct.and the most important 
            thing is give a answer shortly. gen answer and question in thai
            Format as JSON:
            {{
            "question": "your question here?",
            "answer": "your answer here"
            }}

            Return ONLY the JSON object.
            """
            try:
                response = self.llm.invoke(prompt)
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]

                if not json_str:
                    print(f"  ✗ Sample {i+1} skipped: Empty or invalid JSON")
                    continue

                sample = json.loads(json_str)
                question = sample.get("question", "")
                answer = sample.get("answer", "")

                if not question.endswith("?"):
                    print(f"  ⚠️ Sample {i+1} skipped: Question missing '?'")
                    continue

                word_count = len((question + " " + answer).split())
                question_len = len(question)
                answer_len = len(answer)

                cursor.execute(
                    "INSERT INTO raw_data (content, data_type, word_count, question_length, answer_length) VALUES (?, ?, ?, ?, ?)",
                    (json.dumps(sample), topic, word_count, question_len, answer_len)
                )
                created_samples.append(sample)
                print(f"  ✓ Created sample {i+1}/{num_samples}")

            except Exception as e:
                print(f"  ✗ Sample {i+1} error: {e}")

        self.conn.commit()
        return created_samples

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

        with open(filename, "w") as f:
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

    def run_full_pipeline(self, topic="football", num_samples=5, min_quality=7.0):
        print(f"\n🚀 Starting Enhanced Multi-Agent Pipeline for topic: '{topic}'")
        print(f"Samples: {num_samples}, Min Quality: {min_quality}")
        print("="*50)

        print("STEP 1: DATA CREATION")
        raw = self.agent_1_data_creator(topic, num_samples)
        if not raw:
            print("❌ No data generated")
            return

        print("\nSTEP 2: QUALITY FILTER")
        filtered = self.agent_2_quality_filter(min_quality)
        if not filtered:
            print("❌ No samples passed filter")
            return

        print("\nSTEP 3: TRAINING SIMULATION")
        result = self.agent_3_main_trainer()
        if result["status"] == "success":
            print("\n📊 Summary:")
            print(f"✓ {result['count']} samples used")
            print(f"✓ Avg score: {result['average_score']:.2f}")
            print(result['summary'])
        self.export_data("filtered_output.json")
        self.get_statistics()


# ================================
# Usage example
# ================================
if __name__ == "__main__":
    system = MultiAgentSystem(model_name="gemini-2.5-flash", api_key="AIzaSyBzVdrsbkQL7qXmcSZNbVs8kN7jyVfqzF0")
    system.run_full_pipeline(topic="bank of thailand regulation", num_samples=20, min_quality=7.5)