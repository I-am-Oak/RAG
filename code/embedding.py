import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient, IndexModel
from pymongo.server_api import ServerApi
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', title='Vector Store API', description='An API for vectorizing and storing text data')

# Configure logging
logging.basicConfig(filename='../log/embedding.log', level=logging.INFO)

# MongoDB Atlas connection
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME', 'vector_store')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'documents')

if not MONGODB_URI:
    logging.error("MONGODB_URI environment variable not set")
    raise

client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Create vector index
index_model = IndexModel([("vector", "2d")], name="vector_index")

try:
    if "vector_index" not in collection.index_information():
        collection.create_indexes([index_model])
        logging.info("Vector index created successfully")
    else:
        logging.info("Vector index already exists")
except Exception as e:
    logging.error(f"An error occurred while creating the index: {e}")

# Define the namespace
ns = api.namespace('vectors', description='Vector operations')

# Define the input model
chunk_model = api.model('Chunk', {
    'file_name': fields.String(required=True, description='Name of the file'),
    'text': fields.String(required=True, description='Text content')
})

chunks_model = api.model('Chunks', {
    'chunks': fields.List(fields.Nested(chunk_model), required=True, description='List of text chunks')
})

# Function to vectorize text
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

# Function to add metadata and store the vectorized data in MongoDB Atlas Vector Store
def store_vectors(chunks):
    start_time = time.time()
    try:
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            vector = vectorize_text([chunk['text']])[0]
            document = {
                'file_name': chunk['file_name'],
                'text': chunk['text'],
                'vector': vector
            }
            collection.insert_one(document)
            chunk_end_time = time.time()
            logging.info(f"Stored chunk {i+1} of {len(chunks)} in {chunk_end_time - chunk_start_time} seconds")
        end_time = time.time()
        logging.info(f"Successfully stored {len(chunks)} vectors in MongoDB Atlas in {end_time - start_time} seconds")
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error storing vectors in {end_time - start_time} seconds: {e}")
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5002)
