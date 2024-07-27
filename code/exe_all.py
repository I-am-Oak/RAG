import os
import subprocess
import concurrent.futures

def execute_python_file(file_path):
    try:
        print(f"Executing {file_path}...")
        # Run the Python file
        result = subprocess.run(["python", file_path], capture_output=True, text=True)
        
        # Print the output of the Python file execution
        print(f"Output of {file_path}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors in {file_path}:\n{result.stderr}")
    except Exception as e:
        print(f"Failed to execute {file_path}: {e}")

def execute_python_files_in_current_folder(exclude_file):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # List all files in the current directory
    python_files = [
        os.path.join(current_folder, filename)
        for filename in os.listdir(current_folder)
        if filename.endswith(".py") and filename != exclude_file
    ]

    # Execute all Python files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(execute_python_file, python_files)

if __name__ == "__main__":
    # Get the name of this script to exclude it from execution
    exclude_filename = os.path.basename(__file__)
    execute_python_files_in_current_folder(exclude_filename)
