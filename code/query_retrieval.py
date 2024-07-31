import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', title='Query Retrieval API', description='An API for querying and retrieving relevant documents')

# Configure logging
logging.basicConfig(filename='../log/query_retrieval.log', level=logging.INFO)

# Vector Store API URL
VECTOR_STORE_API_URL = os.getenv('VECTOR_STORE_API_URL', 'http://localhost:5002')

# Define the namespace
ns = api.namespace('', description='Query operations')

# Define the input model
query_model = api.model('Query', {
    'query': fields.String(required=True, description='User query')
})

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
                # Send request to Vector Store API's retrieve endpoint
                response = requests.post(f"{VECTOR_STORE_API_URL}/vectors/retrieve", json={'query': query})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Process the response
                    if result.get('status') == 'success':
                        documents = result.get('documents', [])
                        context = ' '.join(doc['content'] for doc in documents)
                        sources = set(doc['metadata']['file_name'] for doc in documents)
                        
                        # Format the final response
                        formatted_response = {
                            'prompt': f"context: {context} \n\nQuestion: {query} \nAnswer: ",
                            'sources': f"{sources}"
                        }
                        return formatted_response, 200
                    else:
                        return {'status': 'error', 'message': 'Error retrieving documents from Vector Store API'}, 500
                else:
                    return {'status': 'error', 'message': 'Error retrieving documents from Vector Store API'}, 500
            else:
                return {'status': 'error', 'message': 'Query is required'}, 400
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return {'status': 'error', 'message': 'Internal Server Error'}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5003)