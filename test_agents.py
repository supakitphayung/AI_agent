# Quick test script - save as test_agents.py
from multi_agent_system import MultiAgentSystem

def quick_test():
    print("🧪 Testing Multi-Agent System...")
    
    # Initialize with your Ollama model
    system = MultiAgentSystem(model_name="llama3.2")  # or whatever model you have
    
    # Test with a simple topic
    result = system.run_full_pipeline(
        topic="football",
        num_samples=3,  # Start small for testing
        min_quality=5.0  # Lower threshold for testing
    )
    
    if result and result['status'] == 'success':
        print("\n🎉 Success! Your multi-agent system is working!")
        
        # Show some sample data
        print("\nSample training data:")
        for i, sample in enumerate(result['sample_data'], 1):
            print(f"\n{i}. Q: {sample['question']}")
            print(f"   A: {sample['answer'][:100]}...")
    
    # Export the data
    system.export_data("test_output.json")

if __name__ == "__main__":
    quick_test()
