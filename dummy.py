import streamlit as st
from work2 import embed_extracted_tables
import os
# Create a dummy folder and add dummy JSON files for testing
folder_name = "dummy_folder"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    # Create a sample JSON file
    sample_data = [{"column1": "value1", "column2": "value2"}]
    with open(f"{folder_name}/sample.json", "w") as f:
        import json
        json.dump(sample_data, f)

st.write("Testing work2 import...")
try:
    result = embed_extracted_tables(folder_name)
    st.write(f"Result: {result}")
except Exception as e:
    st.error(f"Error occurred: {e}")
