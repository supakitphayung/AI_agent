from multi_agent_system import MultiAgentSystem
from google.generativeai import configure

configure(api_key="AIzaSyBzVdrsbkQL7qXmcSZNbVs8kN7jyVfqzF0")

def quick_test():
    print("🧪 Testing Multi-Agent System...")

    system = MultiAgentSystem(model_name="gemini-2.5-flash")

    result = system.run_full_pipeline(
        topic="krungthai bank",
        num_samples=3,
        min_quality=5.0
    )

    if result and result.get('status') == 'success':
        print("\n🎉 Success! Your multi-agent system is working!")
        for i, sample in enumerate(result.get('examples', []), 1):
            print(f"\n{i}. Q: {sample['question']}")
            print(f"   A: {sample['answer'][:100]}...")

    system.export_data("test_output.json")

if __name__ == "__main__":
    quick_test()
