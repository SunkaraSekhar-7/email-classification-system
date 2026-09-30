"""Load and clean the support-ticket dataset."""
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/tickets.csv"


def clean_text(text: str) -> str:
    text = str(text).replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"http\S+", " ", text)        # remove links
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    return text.strip().lower()


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["answer"] = df["answer"].fillna("")
    # subject is missing for ~13% of tickets, so merge it with the body
    df["text"] = (df["subject"] + " " + df["body"]).map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, target: str = "queue", test_size: float = 0.2, seed: int = 42):
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[target])
