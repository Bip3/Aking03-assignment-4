from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse.linalg import svds

app = Flask(__name__)

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents) 

#Rewritten from scratch. Most other code is chatgpt generated for efficiency.
def perform_lsa(X, n_components):
    U, S, VT = svds(X, k=n_components)
    S = np.diag(S)
    return U, S, VT

n_components = 100
U, S, VT = perform_lsa(X, n_components)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
#Done from scratch
def search():
    query = request.form['query']
    query_vec = vectorizer.transform([query]).toarray() 
    query_lsa = np.dot(query_vec, VT.T) 
    similarities = cosine_similarity(query_lsa, U).flatten() 

    top_indices = similarities.argsort()[-5:][::-1]
    top_docs = [(documents[i], similarities[i]) for i in top_indices]

    scores = [score for _, score in top_docs]
    labels = [f'Doc {i + 1}' for i in range(len(scores))]

    return jsonify({
        'top_docs': top_docs,
        'scores': scores,
        'labels': labels
    })

if __name__ == '__main__':
    app.run(debug=True)
