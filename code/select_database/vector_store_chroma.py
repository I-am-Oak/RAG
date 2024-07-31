import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', title='Vector Store API', description='An API for vectorizing, storing, and retrieving text data')

# Configure logging
logging.basicConfig(filename='../../log/vector_store_chroma.log', level=logging.INFO)

# Chroma DB setup
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH')
EMBEDDINGS_MODEL_NAME = os.getenv('EMBEDDINGS_MODEL_NAME')
TARGET_SOURCE_CHUNKS = int(os.getenv('TARGET_SOURCE_CHUNKS', '4'))

# Define the namespace
ns = api.namespace('vectors', description='Vector operations')

# Define the input models
chunk_model = api.model('Chunk', {
    'file_name': fields.String(required=True, description='Name of the file'),
    'text': fields.String(required=True, description='Text content')
})

chunks_model = api.model('Chunks', {
    'chunks': fields.List(fields.Nested(chunk_model), required=True, description='List of text chunks')
})

query_model = api.model('Query', {
    'query': fields.String(required=True, description='Query text')
})

# Initialize Chroma DB components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def initialize_chroma_db():
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            os.makedirs(CHROMA_DB_PATH)
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        return db
    except Exception as e:
        logging.error(f"Error initializing Chroma DB: {e}")
        raise

db = initialize_chroma_db()
retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

# Function to vectorize and store text
def store_vectors(chunks):
    start_time = time.time()
    try:
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{"file_name": chunk['file_name']} for chunk in chunks]
        db.add_texts(texts, metadatas)
        end_time = time.time()
        logging.info(f"Successfully stored {len(chunks)} vectors in Chroma DB in {end_time - start_time} seconds")
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error storing vectors in {end_time - start_time} seconds: {e}")
        raise

# Function to retrieve similar vectors
def retrieve_similar_vectors(query):
    start_time = time.time()
    try:
        docs = retriever.get_relevant_documents(query)
        end_time = time.time()
        logging.info(f"Retrieved similar vectors in {end_time - start_time} seconds")
        return docs
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error retrieving similar vectors in {end_time - start_time} seconds: {e}")
        raise

@ns.route('/store')
class StoreVectors(Resource):
    @api.expect(chunks_model)
    @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
    def post(self):
        """Store vectors for the provided text chunks"""
        try:
            data = request.json
            chunks = data.get('chunks')
            if chunks:
                store_vectors(chunks)
                return {'status': 'success', 'message': f'Stored {len(chunks)} vectors'}, 200
            else:
                api.abort(400, "Chunks are required")
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            api.abort(500, "Internal Server Error")

@ns.route('/retrieve')
class RetrieveVectors(Resource):
    @api.expect(query_model)
    @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
    def post(self):
        """Retrieve relevant documents for the provided query"""
        try:
            data = request.json
            query = data.get('query')
            if query:
                docs = retrieve_similar_vectors(query)
                result = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
                return {'status': 'success', 'documents': result}, 200
            else:
                api.abort(400, "Query is required")
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            api.abort(500, "Internal Server Error")

if __name__ == '__main__':
    app.run(debug=True, port=5002)
