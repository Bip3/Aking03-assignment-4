import matplotlib
# Use a non-interactive backend
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_file
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Preprocess documents and create term-document matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply LSA. CHATGPT IMPLEMENTATION. REPLACED IN FINAL \\todo
n_components = 100  # Number of dimensions for LSA
lsa = TruncatedSVD(n_components=n_components)
X_lsa = lsa.fit_transform(X)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_vec = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vec)
    similarities = cosine_similarity(query_lsa, X_lsa).flatten()

    # Get top 5 document indices
    top_indices = similarities.argsort()[-5:][::-1]
    top_docs = [(documents[i], similarities[i]) for i in top_indices]

    # Generate bar chart
    generate_bar_chart([score for _, score in top_docs])

    return jsonify(top_docs)

def generate_bar_chart(scores):
    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    labels = [f'Doc {i+1}' for i in range(len(scores))]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores, color='cyan')
    plt.ylim(0, 1)  # Assuming similarity scores are between 0 and 1
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Top Documents')
    plt.grid(axis='y')

    # Save the figure to a file
    chart_path = 'static/similarity_chart.png'
    
    try:
        plt.savefig(chart_path)
    except Exception as e:
        print(f"Error saving chart: {e}")
    finally:
        plt.close()  # Close the plot to free memory

@app.route('/chart')
def chart():
    return send_file('static/similarity_chart.png')

if __name__ == '__main__':
    app.run(debug=True)
