import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import openai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="../templates")

CORS(app)
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "embeddings"

if collection_name in chroma_client.list_collections():
    chroma_client.delete_collection(collection_name)
    print(f"Existing collection '{collection_name}' deleted.")

collection = chroma_client.get_or_create_collection("embeddings")
print(f"New collection '{collection_name}' created.")

# Initialize Sentence Transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to chunk text into smaller pieces
def chunk_text(text, max_length=200):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


file_path = '../data/CST Test Data.xlsx'  
df = pd.read_excel(file_path)

# Process the DataFrame and store embeddings in ChromaDB
for index, row in df.iterrows():
    # Combine all columns into one string for embedding
    content = f"Product Name: {row['Product Name']}\nProduct Type: {row['Product Type']}\nIssue Description: {row['Issue Description']}\nResolution Suggestion: {row['Resolution Suggestion']}"
    chunks = chunk_text(content)

    for chunk in chunks:
        # Generate embeddings using SentenceTransformer
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            metadatas=[{"resource_id": index}],
            ids=[f"{index}_{chunks.index(chunk)}"]
        )

print("Embeddings have been successfully stored in ChromaDB.")

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')


def retrieve_context(query, top_n=2):
    # Generate embedding for the query using the same embedding model
    query_embedding = embedding_model.encode(query).tolist()

    # Perform semantic search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n  # Retrieve top N results
    )

    # Flatten the list of documents (results['documents'] is a list of lists)
    flat_documents = [doc for sublist in results['documents'] for doc in sublist]

    # Combine retrieved documents into a single context
    context = "\n".join(flat_documents)
    return context


# Generate response using OpenAI API with retrieved context and query
def generate_response(query, context):
    # Combine query and context into a single prompt
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    # Print the generated prompt
    print("\n--- Generated Context and prompt ---")
    print(prompt)
    print("------------------------\n\n")
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7  # Adjust temperature for creativity vs. determinism
    )

    return response.choices[0].message.content


@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html is in a 'templates' folder

@app.route('/chat', methods=['POST'])
def chat():
    # Ensure the request is JSON formatted
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 400
    
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Retrieve context and generate response dynamically
        context = retrieve_context(query)
        response = generate_response(query, context)
        
        return jsonify({"query": query, "response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app on localhost or any specified host/port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)