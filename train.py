"""Run:  python train.py
Trains the answer-quality model step by step, prints the results and saves it to model.pkl."""
import joblib
from ml_model import make_dataset, build, evaluate

X, y = make_dataset()
print(f"STEP 1: Dataset created -> {len(X)} answers (poor/average/good)")
print("STEP 2: Preprocessing -> TF-IDF converts text into numbers")
print("STEP 3: Split -> 75% training, 25% testing")
print("STEP 4: Training Logistic Regression ...")
model, r = evaluate("logreg")
print("STEP 5: Testing on unseen 25% data")
print(f"   Accuracy : {r['accuracy']:.3f}")
print(f"   Precision: {r['precision']:.3f}")
print(f"   Recall   : {r['recall']:.3f}")
print(f"   F1-score : {r['f1']:.3f}")
print("   Confusion matrix (rows=actual, cols=predicted: poor, average, good):")
print(r["confusion"])
joblib.dump(model, "model.pkl")
print("STEP 6: Trained model saved as model.pkl")
