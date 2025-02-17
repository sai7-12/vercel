from flask import Flask, request, jsonify, render_template
import os
import openai

# Explicitly set the template folder path relative to the `api/` directory
app = Flask(__name__, template_folder="../templates")

# Set your OpenAI API key as an environment variable or directly here
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to generate a response using OpenAI's GPT model
def generate_response(query):
    try:
        # Create a prompt for the chatbot
        prompt = f"You are a helpful assistant. Answer the following question:\n\nQuestion: {query}\n\nAnswer:"
        
        # Call OpenAI API to get a response
        response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7  # Adjust temperature for creativity vs. determinism
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400

    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Generate a response using OpenAI's API
        response = generate_response(query)
        return jsonify({"query": query, "response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
