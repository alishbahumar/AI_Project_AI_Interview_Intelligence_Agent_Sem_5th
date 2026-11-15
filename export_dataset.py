"""Exports the synthetic answer dataset for Weka (ARFF) and Excel/Pandas (CSV).
Weka: Open file -> answers.arff -> Filters: StringToWordVector -> Classify: Logistic / NaiveBayesMultinomial."""
import csv
from ml_model import make_dataset

X, y = make_dataset()
names = ["poor", "average", "good"]
with open("answers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "class"])
    for t, l in zip(X, y): w.writerow([t, names[l]])
with open("answers.arff", "w", encoding="utf-8") as f:
    f.write("@relation interview_answers\n@attribute text string\n@attribute class {poor,average,good}\n@data\n")
    for t, l in zip(X, y): f.write(f"'{t}',{names[l]}\n")
print("Created answers.csv and answers.arff:", len(X), "rows")
