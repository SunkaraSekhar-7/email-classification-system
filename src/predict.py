"""Classify one email and suggest a reply.

Usage:
    python src/predict.py "Subject line. Email body ..."
    python src/predict.py --llm "Subject line. Email body ..."
"""
import argparse
import joblib
from preprocess import clean_text
from respond import Responder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("email", help="email text (subject + body)")
    ap.add_argument("--llm", action="store_true", help="write the reply with Claude (needs API key)")
    args = ap.parse_args()

    text = clean_text(args.email)
    queue_model = joblib.load("models/queue_model.joblib")
    priority_model = joblib.load("models/priority_model.joblib")

    queue = queue_model.predict([text])[0]
    priority = priority_model.predict([text])[0]
    conf = queue_model.predict_proba([text]).max()

    print(f"Category : {queue}  (confidence {conf:.0%})")
    print(f"Priority : {priority}")

    responder = Responder()
    reply = responder.llm_reply(args.email, queue) if args.llm else responder.retrieval_reply(args.email, queue)
    print("\nSuggested reply:\n" + reply)


if __name__ == "__main__":
    main()
