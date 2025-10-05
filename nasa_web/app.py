
from flask import Flask, render_template, request
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import os


app = Flask(__name__)

# Cargar CSV con ruta absoluta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "nasa_pmc_metadata.csv")
df = pd.read_csv(csv_path)
df = df.dropna(subset=["abstract"])

# Cargar modelo de SpaCy
nlp = spacy.load("en_core_web_sm")

# Preprocesamiento y lematización
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

df["clean_abstract"] = df["abstract"].apply(preprocess_text)

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_abstract"].fillna(""))

# Función de búsqueda
def search_articles(query, top_n=5):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_idx].copy()
    results["score"] = (scores[top_idx] * 100).round(2)
    return results

# Función de resaltar palabras
def highlight_keywords(text, keywords):
    if not isinstance(text, str):
        return ""
    keywords = [k.strip().lower() for k in keywords]
    def repl(match):
        return f"<mark>{match.group(0)}</mark>"
    for kw in keywords:
        pattern = re.compile(r"\b{}\b".format(re.escape(kw)), re.IGNORECASE)
        text = pattern.sub(repl, text)
    return text


# Generar WordCloud global
all_text = " ".join(df["clean_abstract"].tolist())
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(all_text)
# Asegurar que la carpeta 'static' existe
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
wordcloud_path = os.path.join(static_dir, "wordcloud.png")
wordcloud.to_file(wordcloud_path)

# Top palabras
word_counts = Counter(all_text.split())
common_words = word_counts.most_common(20)

@app.route("/", methods=["GET", "POST"])
def index():
    top_articles = df.head(3)  # Mostrar 3 más recientes como ejemplo
    if request.method == "POST":
        query = request.form.get("query")
        return search(query)
    return render_template("index.html", top_articles=top_articles, common_words=common_words)

@app.route("/search")
def search(query=None):
    if query is None:
        query = request.args.get("query", "")
    keywords = query.split()
    results = search_articles(query, top_n=10)
    results["highlighted_abstract"] = results["abstract"].apply(lambda x: highlight_keywords(x, keywords))
    return render_template("search_results.html", query=query, results=results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

