from flask import Flask, render_template, request
import pickle
import re
import numpy as np

# Load trained model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.lower()
    return text

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        title = request.form["title"]
        cleaned = clean_text(title)
        
        # Vectorize input
        features = vectorizer.transform([cleaned])
        
        # Predict
        result = model.predict(features)[0]
        prediction = "✅ Real News" if result == 1 else "❌ Fake News"
        
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=False)

