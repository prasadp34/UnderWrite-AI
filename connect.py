import streamlit as st
import subprocess
import sys
import os

# Title of the app
st.title("Select the loan type...!")



# Navigation options
option = st.selectbox("Choose a file to execute:", ("business.py", "personal.py"))

# Paths to the files (update these paths to the correct locations on your system)
business_file_path = r"C:\Users\Prasad\Desktop\Cognizant\business.py"  # Update this path
personal_file_path = r"C:\Users\Prasad\Desktop\Cognizant\personal.py"  # Update this path

# Function to execute a Python file
def run_python_file(file_path):
    try:
        # Use a command to run a separate Streamlit process
        command = [sys.executable, "-m", "streamlit", "run", file_path]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Display the output in the Streamlit app
        st.write("### Output")
        st.text(result.stdout)
        if result.stderr:
            st.write("### Errors")
            st.error(result.stderr)
    except Exception as e:
        st.error(f"An error occurred while executing the script: {e}")

# Execute file content
if option == "business.py":
    st.write("### Execute business.py")
    if st.button("Run business"):
        run_python_file(business_file_path)

elif option == "personal.py":
    st.write("### Execute personal.py")
    if st.button("Run personal"):
        run_python_file(personal_file_path)
