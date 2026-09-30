"""Train and evaluate the ticket classifiers (queue + priority)."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import make_pipeline

from preprocess import load_data, split_data


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        # LinearSVC scored best in my comparison; calibration gives us probabilities
        CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3),
    )


def run(target: str, df, results: dict):
    train, test = split_data(df, target=target)
    model = build_model()
    model.fit(train["text"], train[target])
    pred = model.predict(test["text"])

    acc = accuracy_score(test[target], pred)
    macro_f1 = f1_score(test[target], pred, average="macro")
    print(f"\n=== {target} ===  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}")
    print(classification_report(test[target], pred, zero_division=0))
    results[target] = {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
                       "train_size": len(train), "test_size": len(test)}

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(test[target], pred, xticks_rotation=45,
                                            ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {target}")
    plt.tight_layout()
    plt.savefig(f"reports/confusion_{target}.png", dpi=120)
    plt.close(fig)

    joblib.dump(model, f"models/{target}_model.joblib")
    return train


def main():
    df = load_data()
    print("Loaded", len(df), "tickets")
    results = {}
    train_df = run("queue", df, results)
    run("priority", df, results)
    # keep the training tickets so respond.py can retrieve similar past replies
    train_df[["text", "queue", "language", "answer"]].to_pickle("models/reply_bank.pkl")
    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved models to models/, reports to reports/")


if __name__ == "__main__":
    main()
