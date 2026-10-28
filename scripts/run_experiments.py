from optimization.benchmark import benchmark
from knowledge_assistant.evaluation import evaluate as knowledge
from operations_agent.evaluation import evaluate as agent

if __name__ == "__main__":
    benchmark()
    knowledge()
    agent()
