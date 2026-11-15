"""ML module: classifies answer quality (0=poor, 1=average, 2=good) from text.
Dataset is SYNTHETIC (generated from keyword pools with label noise) because
no real labelled interview data is available; see report for limitations."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

TECH = ("algorithm complexity optimize index normalization overfitting regularization hash recursion "
        "tradeoff scalable validation example because therefore implemented tested measured").split()
VAGUE = "thing stuff maybe kind basically somehow good nice like whatever dont know".split()
NEUTRAL = "the is a of to and in it we use data system code problem".split()
MIX = {0: (1, 5, 8), 1: (4, 8, 4), 2: (8, 8, 1)}  # (tech, neutral, vague) counts


def make_answer(label, rng, noise=0.2):
    if rng.random() < noise:                       # label noise: borderline answers
        label = int(np.clip(label + rng.choice([-1, 1]), 0, 2))
    t, n, v = MIX[label]
    words = list(rng.choice(TECH, t)) + list(rng.choice(NEUTRAL, n)) + list(rng.choice(VAGUE, v))
    rng.shuffle(words)
    return " ".join(words)


def make_dataset(n=900, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = [make_answer(int(l), rng) for l in y]
    return X, y


def build(kind="logreg"):
    clf = LogisticRegression(max_iter=1000) if kind == "logreg" else MultinomialNB()
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), clf)


def evaluate(kind="logreg", seed=42):
    X, y = make_dataset(seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    m = build(kind).fit(Xtr, ytr)
    p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
    return m, {"accuracy": accuracy_score(yte, p), "precision": pr, "recall": rc, "f1": f1,
               "confusion": confusion_matrix(yte, p)}
