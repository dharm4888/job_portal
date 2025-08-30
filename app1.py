from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('cleaned_jobs.csv')

# Fill missing values to avoid errors
df['title'] = df['title'].fillna('')
df['link'] = df['link'].fillna('#')
df['salary'] = df['salary'].fillna(0.0)
df['country'] = df['country'].fillna('Unknown')

# Vectorize job titles
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['title'])

# Recommendation function
def recommend_jobs(user_query, top_n=5):
    if not user_query or not isinstance(user_query, str):
        return pd.DataFrame(columns=['title', 'link', 'salary', 'country'])
    
    query_vec = vectorizer.transform([user_query.lower()])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    
    recommendations = df.iloc[top_indices][['title', 'link', 'salary', 'country']].copy()
    recommendations['similarity_score'] = sim_scores[top_indices]
    return recommendations

# API endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query')
        top_n = data.get('top_n', 5)
        
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        if not isinstance(top_n, int) or top_n <= 0:
            return jsonify({'error': 'top_n must be a positive integer'}), 400
        
        recs = recommend_jobs(query, min(top_n, len(df)))
        if recs.empty:
            return jsonify({'message': 'No recommendations found for the given query'}), 200
        
        response = recs.to_dict(orient='records')
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# UI Route
@app.route('/')
def home():
    html_template = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>Job Recommendation System</title>
        <style>
            body {{font-family: Arial, sans-serif;background-color: #f4f4f4;margin: 0;padding: 0;
            display: flex;justify-content: center;align-items: center;min-height: 100vh;}}
            .container {{background-color: white;padding: 20px;border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);width: 80%;max-width: 600px;text-align: center;}}
            h1 {{color: #333;margin-bottom: 20px;}}
            .form-group {{margin-bottom: 15px;}}
            input[type='text'], input[type='number'] {{padding: 8px;width: 70%;max-width: 300px;
            margin-right: 10px;border: 1px solid #ddd;border-radius: 4px;}}
            input[type='submit'] {{padding: 8px 15px;background-color: #007BFF;color: white;
            border: none;border-radius: 4px;cursor: pointer;}}
            input[type='submit']:hover {{background-color: #0056b3;}}
            #result {{margin-top: 20px;text-align: left;white-space: pre-wrap;max-height: 300px;
            overflow-y: auto;border: 1px solid #ddd;padding: 10px;background-color: #f9f9f9;}}
            .footer {{margin-top: 20px;font-size: 0.9em;color: #666;}}
        </style>
    </head>
    <body>
        <div class='container'>
            <h1>Job Recommendation System</h1>
            <form id='recommendForm'>
                <div class='form-group'>
                    <input type='text' id='query' name='query' placeholder='Enter skills (e.g., python developer)' required>
                    <input type='number' id='top_n' name='top_n' placeholder='Number of recommendations (e.g., 5)' min='1' value='5'>
                    <input type='submit' value='Recommend'>
                </div>
            </form>
            <div id='result'></div>
            <div class='footer'>Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        <script>
            document.getElementById('recommendForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const query = document.getElementById('query').value;
                const top_n = parseInt(document.getElementById('top_n').value);
                const response = await fetch('/recommend', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query: query, top_n: top_n}})
                }});
                const result = await response.json();
                const resultDiv = document.getElementById('result');
                if (response.ok) {{
                    if (result.message) {{
                        resultDiv.textContent = result.message;
                    }} else {{
                        resultDiv.innerHTML = '<h3>Recommendations:</h3>' +
                        result.map(job => `<div>
                            <strong>${{job.title || 'N/A'}}</strong> (${{job.country || 'Unknown'}}) - $${{(job.salary || 0).toFixed(2)}}<br>
                            <a href='${{job.link || '#'}}' target='_blank'>View Job</a><br>
                            Similarity: ${{job.similarity_score.toFixed(2)}}<br><br>
                        </div>`).join('');
                    }}
                }} else {{
                    resultDiv.textContent = `Error: ${{result.error || 'Unknown error'}}`;
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_template

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
