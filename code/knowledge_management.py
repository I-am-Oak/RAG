import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Knowledge Management API', description='A simple Flask API for document processing')

# Configure logging
log_dir = '../log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, 'knowledge_management.log'), level=logging.INFO)

# Define the namespace
ns = api.namespace('documents', description='Document processing operations')

# Define the input model
folder_path_model = api.model('FolderPath', {
    'folder_path': fields.String(required=True, description='Path to the folder containing PDF documents')
})

# Function to create a DirectoryLoader for PDF files
def create_directory_loader(directory_path):
    return PyMuPDFLoader(directory_path)

# Function to process pdf documents: text splitting and chunking
def process_documents(folder_path):
    start_time = time.time()
    try:
        documents = []
        for file in os.listdir(folder_path):
            if file.endswith('.pdf'):
                file_start_time = time.time()
                file_path = os.path.join(folder_path, file)
                loader = create_directory_loader(file_path)
                doc = loader.load()
                documents.extend(doc)
                file_end_time = time.time()
                logging.info(f"Processed file {file} in {file_end_time - file_start_time} seconds")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=30
        )
        chunks = text_splitter.split_documents(documents)

        # Convert Document objects to dictionaries
        processed_chunks = []
        for chunk in chunks:
            try:
                processed_chunks.append({
                    'file_name': chunk.metadata.get('source', 'Unknown'),
                    'text': chunk.page_content
                })
            except AttributeError as e:
                logging.error(f"Error processing chunk: {e}")
                logging.error(f"Chunk data: {chunk}")

        end_time = time.time()
        logging.info(f"Total processing time for folder {folder_path}: {end_time - start_time} seconds")
        return processed_chunks
    except Exception as e:
        end_time = time.time()
        logging.error(f"Error processing documents in {end_time - start_time} seconds: {e}")
        return []

@ns.route('/process_documents')
class ProcessDocuments(Resource):
    @api.expect(folder_path_model)
    @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
    def post(self):
        """Process documents in the specified folder"""
        try:
            data = request.json
            folder_path = data.get('folder_path')
            if folder_path:
                logging.info(f"Processing documents in folder: {folder_path}")
                chunks = process_documents(folder_path)
                logging.info(f"Successfully processed {len(chunks)} chunks")
                return {'status': 'success', 'chunks': chunks}, 200
            else:
                logging.warning("Request received without folder path")
                api.abort(400, "Folder path is required")
        except Exception as e:
            logging.error(f"Error processing request: {e}", exc_info=True)
            api.abort(500, "Internal Server Error")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
