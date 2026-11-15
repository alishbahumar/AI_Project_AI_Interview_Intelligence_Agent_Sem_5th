"""Model-based, utility-driven interview agent.
Loop: PLAN (A*) -> ASK first question -> SENSE answer -> ML scores it ->
UPDATE skill beliefs -> REPLAN. Output is decision *support*, not a hiring decision."""
import os
import joblib
import numpy as np
from question_bank import QUESTIONS, TOPICS
from search import astar, evidence_of
from ml_model import evaluate, make_answer


class InterviewAgent:
    def __init__(self, time_budget=25, alpha=0.6, model=None):
        self.qs = QUESTIONS
        self.budget, self.alpha = time_budget, alpha
        self.model = model or (joblib.load("model.pkl") if os.path.exists("model.pkl") else evaluate("logreg")[0])
        self.skill = {t: 2.0 for t in TOPICS}      # belief state (1..3)
        self.ev = {t: 0.0 for t in TOPICS}
        self.asked, self.log, self.nodes = [], [], 0

    def run(self, answer_fn, verbose=True):
        while True:
            used = sum(self.qs[i]["time"] for i in self.asked)
            plan, cost, n = astar(self.qs, self.skill, self.ev, set(self.asked), self.budget - used)
            self.nodes += n
            if plan is None or not plan:
                break
            q = self.qs[plan[0]]
            ans = answer_fn(q)
            cls = int(self.model.predict([ans])[0])           # 0,1,2
            self.ev[q["topic"]] += evidence_of(q, self.skill)
            self.skill[q["topic"]] = (1 - self.alpha) * self.skill[q["topic"]] + self.alpha * (cls + 1)
            self.asked.append(q["id"])
            self.log.append((q["topic"], q["difficulty"], cls))
            if verbose:
                print(f"Q{len(self.asked)} [{q['topic']}, d={q['difficulty']}] {q['text']}\n   -> predicted quality: {['poor','average','good'][cls]}")
        return self.report()

    def report(self):
        overall = float(np.mean(list(self.skill.values())))
        rec = "Strong" if overall >= 2.3 else "Moderate" if overall >= 1.6 else "Needs improvement"
        return {"skills": {k: round(v, 2) for k, v in self.skill.items()},
                "overall": round(overall, 2), "summary": rec,
                "questions_asked": len(self.asked), "minutes_used": sum(self.qs[i]["time"] for i in self.asked)}


def simulated_candidate(profile, seed=0):
    rng = np.random.default_rng(seed)
    def fn(q):
        lab = int(np.clip(round(profile[q["topic"]] - 1 + rng.normal(0, 0.4) - (q["difficulty"] - 2) * 0.4), 0, 2))
        return make_answer(lab, rng)
    return fn


def interactive_candidate(q):
    return input(f"\n[{q['topic']}] {q['text']}\nYour answer: ")
