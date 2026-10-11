import pandas as pd

fake = pd.read_csv("archive/Fake.csv")
true = pd.read_csv("archive/True.csv")
fake["label"] = 1   # 1 = fake
true["label"] = 0   # 0 = real

df = pd.concat([fake, true])[["text", "label"]].dropna()

# Real articles start with "CITY (Reuters) -", which would let the model
# cheat by spotting that tag. Remove it.
df["text"] = df["text"].str.replace(r"^.{0,60}\(Reuters\)\s*-\s*", "", regex=True)

df = df[df["text"].str.strip().str.len() > 0]
df = df.drop_duplicates().sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("FakeNews.csv", index=False)
print("Saved FakeNews.csv with", len(df), "rows")
print(df["label"].value_counts())