import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
import torch
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', title='Query Retrieval API', description='An API for querying and retrieving relevant documents')

# Configure logging
logging.basicConfig(filename='../log/query_retrieval.log', level=logging.INFO)

# MongoDB Atlas connection
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME', 'vector_store')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')

if not MONGODB_URI:
    logging.error("MONGODB_URI environment variable not set")
    raise

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Define the namespace
ns = api.namespace('', description='Query operations')

# Define the input model
query_model = api.model('Query', {
    'query': fields.String(required=True, description='User query')
})

def vectorize_text(texts):
    start_time = time.time()
    try:
        model_name = "BAAI/bge-small-en-v1.5"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        vectors = embeddings.embed_documents(texts)
        end_time = time.time()
        logging.info(f"Vectorization completed in {end_time - start_time} seconds using device: {device}")
        return vectors
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error vectorizing text in {end_time - start_time} seconds: {e}")
        raise

# Function to create embeddings using the BGE model
def create_embedding(text):
    start_time = time.time()
    try:
        vector = vectorize_text([text])[0]
        end_time = time.time()
        logging.info(f"Embedding created in {end_time - start_time} seconds")
        return vector
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error creating embedding in {end_time - start_time} seconds: {e}")
        return None

# Function to vector search in the MongoDB Vector store to retrieve 3 relevant documents
def vector_search(query_vector, top_k=2):
    start_time = time.time()
    try:
        documents = list(collection.find())
        document_vectors = [np.array(doc['vector']) for doc in documents]
        query_vector = np.array(query_vector).reshape(1, -1)
        similarities = cosine_similarity(query_vector, document_vectors)
        top_k_indices = similarities.argsort()[0][-top_k:][::-1]
        top_k_documents = [documents[i] for i in top_k_indices]
        end_time = time.time()
        logging.info(f"Vector search completed in {end_time - start_time} seconds")
        return top_k_documents
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error performing vector search in {end_time - start_time} seconds: {e}")
        return []

# Function to template the documents and user's query as a prompt
def create_prompt(query, documents):
    start_time = time.time()
    try:
        context = '\n'.join([doc['text'] for doc in documents])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        end_time = time.time()
        logging.info(f"Prompt creation completed in {end_time - start_time} seconds")
        return prompt
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error creating prompt in {end_time - start_time} seconds: {e}")
        return None

@ns.route('/query')
class QueryResource(Resource):
    @api.expect(query_model)
    @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
    def post(self):
        """Query retrieval"""
        try:
            data = request.json
            query = data.get('query')

            if query:
                query_vector = create_embedding(query)
                if query_vector:
                    documents = vector_search(query_vector)
                    if documents:
                        prompt = create_prompt(query, documents)
                        return {'status': 'success', 'prompt': prompt}, 200
                    else:
                        return {'status': 'error', 'message': 'No documents found'}, 400
                else:
                    return {'status': 'error', 'message': 'Error creating embedding'}, 500
            else:
                return {'status': 'error', 'message': 'Query is required'}, 400
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return {'status': 'error', 'message': 'Internal Server Error'}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5003)
