"""Generate a reply for an email.

Default (no API key needed): retrieve the most similar past ticket from the same
queue and reuse its reply.
Optional (--llm): ask Claude to write a reply using similar past replies as examples.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from preprocess import clean_text


class Responder:
    def __init__(self):
        self.bank = pd.read_pickle("models/reply_bank.pkl")
        self.bank = self.bank[self.bank["answer"].str.len() > 0].reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.bank["text"])

    def similar_replies(self, text: str, queue: str, k: int = 3):
        mask = (self.bank["queue"] == queue).values
        sims = linear_kernel(self.vec.transform([clean_text(text)]), self.matrix).ravel()
        sims[~mask] = -1
        top = sims.argsort()[::-1][:k]
        return self.bank.iloc[top]["answer"].tolist(), sims[top].tolist()

    def retrieval_reply(self, text: str, queue: str) -> str:
        replies, _ = self.similar_replies(text, queue, k=1)
        return replies[0]

    def llm_reply(self, text: str, queue: str) -> str:
        from anthropic import Anthropic  # needs ANTHROPIC_API_KEY in the environment
        examples, _ = self.similar_replies(text, queue, k=3)
        example_block = "\n\n".join(f"Example reply {i+1}:\n{e}" for i, e in enumerate(examples))
        prompt = (
            f"You are a customer support agent for the '{queue}' team.\n"
            f"Write a polite, concise reply to the email below, in the same language as the email.\n\n"
            f"{example_block}\n\nEmail:\n{text}\n\nReply:"
        )
        client = Anthropic()
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
