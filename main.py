"""Demo:  python main.py --simulate   |   python main.py --interactive"""
import argparse, json
from agent import InterviewAgent, simulated_candidate, interactive_candidate

ap = argparse.ArgumentParser()
ap.add_argument("--interactive", action="store_true")
ap.add_argument("--simulate", action="store_true")
ap.add_argument("--budget", type=int, default=25)
a = ap.parse_args()
agent = InterviewAgent(time_budget=a.budget)
if a.interactive:
    fn = interactive_candidate
else:
    fn = simulated_candidate({"Python": 3, "DataStructures": 2.5, "MachineLearning": 1.5, "Databases": 2, "Behavioral": 3})
print(json.dumps(agent.run(fn), indent=2))
