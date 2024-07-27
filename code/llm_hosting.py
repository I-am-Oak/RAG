import logging
import time
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import torch
from langchain_community.llms import CTransformers
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
api = Api(app, version='1.0', title='LLM Response Generation API', description='An API for generating responses using a local LLM model')

# Configure logging
logging.basicConfig(filename='../log/llm_hosting.log', level=logging.INFO)

# Define the namespace
ns = api.namespace('llm', description='LLM operations')

# Define the input model
prompt_model = api.model('Prompt', {
    'prompt': fields.String(required=True, description='User prompt')
})

# Initialize language model and embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    start_time = time.time()
    logging.info("Initializing language model...")
    model = CTransformers(model="../data/mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", max_new_tokens=1048, temperature=0.1, device=device)
    end_time = time.time()
    logging.info(f"Language model initialized successfully in {end_time - start_time:.2f} seconds")
except Exception as e:
    logging.error(f"Error initializing language model: {e}")
    exit(1)

# Function to generate a response using the LLM model
def generate_response(prompt, max_length=100):
    try:
        start_time = time.time()
        result = model.generate([prompt], max_length=max_length)  # Wrap the prompt in a list
        response = result.generations[0][0].text  # Extract the text from the result object
        end_time = time.time()
        logging.info(f"Response generated successfully in {end_time - start_time:.2f} seconds")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

@ns.route('/generate_response')
class GenerateResponse(Resource):
    @api.expect(prompt_model)
    @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
    def post(self):
        """Generate response using the LLM model"""
        try:
            data = request.json
            prompt = data.get('prompt')

            if prompt:
                logging.info(f"Received prompt: {prompt}")      #prompt templace can be added
                response = generate_response(prompt)
                if response:
                    logging.info("Response generated successfully")
                    return {'status': 'success', 'response': response}, 200
                else:
                    logging.error("Error generating response")
                    return {'status': 'error', 'message': 'Error generating response'}, 500
            else:
                logging.warning("Prompt is missing in the request")
                return {'status': 'error', 'message': 'Prompt is required'}, 400
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return {'status': 'error', 'message': 'Internal Server Error'}, 500

if __name__ == '__main__':
    logging.info("Starting the Flask application...")
    app.run(host='0.0.0.0', debug=True, port=5004)


#meta-llama3.1

# import logging
# import time
# from flask import Flask, request, jsonify
# from flask_restx import Api, Resource, fields
# import torch
# import transformers

# app = Flask(__name__)
# api = Api(app, version='1.0', title='LLM Response Generation API', description='An API for generating responses using a local LLM model')

# # Configure logging
# logging.basicConfig(filename='../log/llm_hosting.log', level=logging.INFO)

# # Define the namespace
# ns = api.namespace('llm', description='LLM operations')

# # Define the input model
# prompt_model = api.model('Prompt', {
#     'prompt': fields.String(required=True, description='User prompt')
# })

# # Initialize language model and embeddings
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# try:
#     start_time = time.time()
#     logging.info("Initializing language model...")
#     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",
#     )
#     end_time = time.time()
#     logging.info(f"Language model initialized successfully in {end_time - start_time:.2f} seconds")
# except Exception as e:
#     logging.error(f"Error initializing language model: {e}")
#     exit(1)

# # Function to generate a response using the LLM model
# def generate_response(prompt, max_length=250):
#     try:
#         start_time = time.time()
#         response = pipeline(
#             prompt,
#             max_new_tokens=max_length,
#         )[0]["generated_text"]
#         end_time = time.time()
#         logging.info(f"Response generated successfully in {end_time - start_time:.2f} seconds")
#         return response
#     except Exception as e:
#         logging.error(f"Error generating response: {e}")
#         return None

# @ns.route('/generate_response')
# class GenerateResponse(Resource):
#     @api.expect(prompt_model)
#     @api.doc(responses={200: 'Success', 400: 'Validation Error', 500: 'Internal Server Error'})
#     def post(self):
#         """Generate response using the LLM model"""
#         try:
#             data = request.json
#             prompt = data.get('prompt')

#             if prompt:
#                 logging.info(f"Received prompt: {prompt}")
#                 response = generate_response(prompt)
#                 if response:
#                     logging.info("Response generated successfully")
#                     return {'status': 'success', 'response': response}, 200
#                 else:
#                     logging.error("Error generating response")
#                     return {'status': 'error', 'message': 'Error generating response'}, 500
#             else:
#                 logging.warning("Prompt is missing in the request")
#                 return {'status': 'error', 'message': 'Prompt is required'}, 400
#         except Exception as e:
#             logging.error(f"Error processing request: {e}")
#             return {'status': 'error', 'message': 'Internal Server Error'}, 500

# if __name__ == '__main__':
#     logging.info("Starting the Flask application...")
#     app.run(host='0.0.0.0', debug=True, port=5004)

