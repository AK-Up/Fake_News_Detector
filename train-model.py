import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# Load datasets
true_df = pd.read_csv("true.csv")
fake_df = pd.read_csv("fake.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
df = df[["title", "label"]].dropna().reset_index(drop=True)

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned_title"] = df["title"].apply(clean_text)

# Check balance
print("Before balancing:", df["label"].value_counts())

# Balance the dataset
df_fake = df[df["label"] == 0]
df_real = df[df["label"] == 1]

df_real_downsampled = resample(df_real, n_samples=23481, random_state=42)

# 👇 Instead, yeh correct wala code lagao:
min_count = min(len(df_fake), len(df_real))

df_fake_downsampled = resample(df_fake,
                               replace=False,
                               n_samples=min_count,
                               random_state=42)

df_real_downsampled = resample(df_real,
                               replace=False,
                               n_samples=min_count,
                               random_state=42)


df_balanced = pd.concat([df_fake_downsampled, df_real_downsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("After balancing:", df_balanced["label"].value_counts())

# TF-IDF with bigrams
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = tfidf.fit_transform(df_balanced["cleaned_title"]).toarray()
y = df_balanced["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save model and vectorizer
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
