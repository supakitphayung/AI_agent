# Multi-Agent System with Ollama
# Install required packages first:
# pip install langchain-ollama pandas sqlite3

import json
import sqlite3
import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any
import os

class MultiAgentSystem:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing data between agents"""
        self.conn = sqlite3.connect('agent_data.db')
        cursor = self.conn.cursor()
        
        # Table for raw data from Agent 1
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                data_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for filtered data from Agent 2
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filtered_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_id INTEGER,
                content TEXT,
                quality_score REAL,
                filter_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (raw_id) REFERENCES raw_data (id)
            )
        ''')
        
        self.conn.commit()
    
    def agent_1_data_creator(self, topic: str, num_samples: int = 5) -> List[Dict]:
        """Agent 1: Creates synthetic training data"""
        print(f"🤖 Agent 1: Creating {num_samples} data samples about '{topic}'...")
        
        # Create samples one by one to avoid JSON parsing issues
        created_samples = []
        cursor = self.conn.cursor()
        
        for i in range(num_samples):
            prompt = f"""
            Create 1 question-answer pair about {topic}.
            Keep answers under 200 words.
            Format as valid JSON:
            {{"question": "your question here", "answer": "your answer here"}}
            
            Return ONLY the JSON object, nothing else.
            """
            
            try:
                response = self.llm.invoke(prompt)
                
                # Clean up response - remove extra text before/after JSON
                response = response.strip()
                
                # Find JSON object in response
                start = response.find('{')
                end = response.rfind('}') + 1
                
                if start != -1 and end > start:
                    json_str = response[start:end]
                    sample = json.loads(json_str)
                    
                    # Validate that it has required fields
                    if 'question' in sample and 'answer' in sample:
                        cursor.execute(
                            "INSERT INTO raw_data (content, data_type) VALUES (?, ?)",
                            (json.dumps(sample), topic)
                        )
                        sample_id = cursor.lastrowid
                        created_samples.append({
                            'id': sample_id,
                            'content': sample,
                            'data_type': topic
                        })
                        print(f"  ✓ Created sample {i+1}/{num_samples}")
                    else:
                        print(f"  ✗ Sample {i+1} missing required fields")
                else:
                    print(f"  ✗ Sample {i+1} no valid JSON found")
                    
            except json.JSONDecodeError as e:
                print(f"  ✗ Sample {i+1} JSON parse error: {e}")
                print(f"    Response: {response[:100]}...")
            except Exception as e:
                print(f"  ✗ Sample {i+1} error: {e}")
        
        self.conn.commit()
        print(f"✅ Agent 1: Successfully created {len(created_samples)} samples")
        return created_samples
    
    def agent_2_quality_filter(self, min_quality_score: float = 7.0) -> List[Dict]:
        """Agent 2: Filters data based on quality criteria"""
        print(f"🔍 Agent 2: Filtering data with minimum quality score {min_quality_score}...")
        
        # Get all raw data
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, data_type FROM raw_data")
        raw_samples = cursor.fetchall()
        
        filtered_samples = []
        
        for sample_id, content_str, data_type in raw_samples:
            try:
                content = json.loads(content_str)
                
                # Quality evaluation prompt
                eval_prompt = f"""
                Evaluate this Q&A pair for quality on a scale of 1-10:
                Question: {content.get('question', '')}
                Answer: {content.get('answer', '')}
                
                Consider:
                - Clarity and specificity of question
                - Accuracy and completeness of answer
                - Educational value
                - Proper grammar and formatting
                
                Respond with only a number (1-10) and brief reason.
                Format: "Score: X, Reason: brief explanation"
                """
                
                eval_response = self.llm.invoke(eval_prompt)
                
                # Parse quality score
                try:
                    score_line = eval_response.split('\n')[0]
                    score = float(score_line.split(':')[1].split(',')[0].strip())
                    reason = eval_response.split('Reason:')[1].strip() if 'Reason:' in eval_response else "No reason provided"
                except:
                    score = 5.0  # Default score if parsing fails
                    reason = "Parsing failed"
                
                if score >= min_quality_score:
                    cursor.execute(
                        "INSERT INTO filtered_data (raw_id, content, quality_score, filter_reason) VALUES (?, ?, ?, ?)",
                        (sample_id, content_str, score, reason)
                    )
                    filtered_samples.append({
                        'id': sample_id,
                        'content': content,
                        'quality_score': score,
                        'reason': reason
                    })
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
        
        self.conn.commit()
        print(f"✅ Agent 2: Filtered {len(filtered_samples)} quality samples")
        return filtered_samples
    
    def agent_3_main_trainer(self) -> Dict:
        """Agent 3: Uses filtered data for training/fine-tuning simulation"""
        print("🎯 Agent 3: Preparing training data and simulating training...")
        
        # Get filtered data
        cursor = self.conn.cursor()
        cursor.execute("SELECT content, quality_score FROM filtered_data")
        training_data = cursor.fetchall()
        
        if not training_data:
            return {"status": "error", "message": "No filtered data available for training"}
        
        # Simulate training process
        training_samples = []
        total_quality = 0
        
        for content_str, quality_score in training_data:
            content = json.loads(content_str)
            training_samples.append(content)
            total_quality += quality_score
        
        avg_quality = total_quality / len(training_data)
        
        # Generate training summary
        summary_prompt = f"""
        Analyze this training dataset of {len(training_samples)} Q&A pairs.
        Average quality score: {avg_quality:.2f}
        
        Sample questions: {[sample['question'][:50] + '...' for sample in training_samples[:3]]}
        
        Provide a brief training summary including:
        - Dataset strengths
        - Potential areas for improvement
        - Recommended next steps
        """
        
        training_analysis = self.llm.invoke(summary_prompt)
        
        result = {
            "status": "success",
            "training_samples": len(training_samples),
            "average_quality": avg_quality,
            "analysis": training_analysis,
            "sample_data": training_samples[:2]  # Show first 2 samples
        }
        
        print(f"✅ Agent 3: Training simulation complete")
        return result
    
    def run_full_pipeline(self, topic: str, num_samples: int = 5, min_quality: float = 7.0):
        """Run the complete multi-agent pipeline"""
        print(f"\n🚀 Starting Multi-Agent Pipeline for topic: '{topic}'\n")
        
        # Step 1: Data Creation
        raw_data = self.agent_1_data_creator(topic, num_samples)
        if not raw_data:
            print("❌ Pipeline failed at data creation step")
            return
        
        # Step 2: Quality Filtering
        filtered_data = self.agent_2_quality_filter(min_quality)
        if not filtered_data:
            print("❌ Pipeline failed: No data passed quality filter")
            return
        
        # Step 3: Training Simulation
        training_result = self.agent_3_main_trainer()
        
        # Final Summary
        print(f"\n📊 Pipeline Summary:")
        print(f"Raw samples created: {len(raw_data)}")
        print(f"Samples passed filter: {len(filtered_data)}")
        print(f"Training status: {training_result['status']}")
        
        if training_result['status'] == 'success':
            print(f"Average quality score: {training_result['average_quality']:.2f}")
            print(f"\nTraining Analysis:\n{training_result['analysis']}")
        
        return training_result
    
    def export_data(self, filename: str = "training_data.json"):
        """Export filtered data for external use"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content, quality_score FROM filtered_data")
        data = cursor.fetchall()
        
        export_data = []
        for content_str, quality_score in data:
            content = json.loads(content_str)
            content['quality_score'] = quality_score
            export_data.append(content)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Exported {len(export_data)} samples to {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = MultiAgentSystem(model_name="llama3.2")  # Change model if needed
    
    # Run the pipeline
    result = system.run_full_pipeline(
        topic="Python programming basics",
        num_samples=8,
        min_quality=6.0
    )
    
    # Export the data
    system.export_data("python_training_data.json")