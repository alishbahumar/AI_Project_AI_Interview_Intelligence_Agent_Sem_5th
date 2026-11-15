"""Question bank: 5 topics x 3 difficulty levels (1=easy, 3=hard)."""
TOPICS = ["Python", "DataStructures", "MachineLearning", "Databases", "Behavioral"]
TIME = {1: 2, 2: 3, 3: 5}  # minutes per difficulty

_TEXT = {
    "Python": ["What is a list vs a tuple?", "Explain decorators and generators.", "How does the GIL affect multithreading?"],
    "DataStructures": ["What is a stack vs a queue?", "Explain how a hash table handles collisions.", "Compare BST, AVL and heap complexity."],
    "MachineLearning": ["What is overfitting?", "Explain precision vs recall.", "How do regularization and cross-validation work together?"],
    "Databases": ["What is a primary key?", "Explain normalization and indexes.", "How do transactions, isolation levels and deadlocks interact?"],
    "Behavioral": ["Tell me about yourself.", "Describe a conflict in a team project.", "Describe a failure and what you changed afterwards."],
}

QUESTIONS = []
for t in TOPICS:
    for d in (1, 2, 3):
        QUESTIONS.append({"id": len(QUESTIONS), "topic": t, "difficulty": d,
                          "time": TIME[d], "text": _TEXT[t][d - 1]})
