"""Empirical evaluation: ML metrics, A* vs BFS, runtime, failure modes."""
import time
from ml_model import evaluate, make_answer
from search import astar, bfs
from question_bank import QUESTIONS, TOPICS
import numpy as np

for kind in ("logreg", "nb"):
    _, r = evaluate(kind)
    print(f"{kind}: acc={r['accuracy']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}\n{r['confusion']}")

skill = {t: 2.0 for t in TOPICS}; ev = {t: 0.0 for t in TOPICS}
for name, f in (("A*", astar), ("BFS", bfs)):
    t0 = time.perf_counter(); plan, cost, n = f(QUESTIONS, skill, ev, set(), 25)
    print(f"{name}: plan={plan} cost={cost:.2f} nodes_expanded={n} time={time.perf_counter()-t0:.3f}s")

# Failure mode: a good-but-short answer and keyword stuffing
m, _ = evaluate("logreg")
print("Short answer  ->", m.predict(["use hash"])[0])
rng = np.random.default_rng(1)
print("Keyword stuffing ->", m.predict(["algorithm complexity optimize index hash recursion tradeoff validation " * 2])[0], "(gaming risk)")
