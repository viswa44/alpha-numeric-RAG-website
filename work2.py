import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import pandas as pd
from google.cloud import storage
import tarfile
import shutil
import speech_recognition as sr  # For voice-to-text transcription
from extract_tables_json import extract_tables_from_pdfs

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Set up Google Cloud Storage client
storage_client = storage.Client()
bucket_name = "rag-bucket452"  # Replace with your GCS bucket name

# Compress a directory into a .tar.gz archive
def compress_directory_to_tar(directory_path, tar_file_path):
    """Compresses a directory into a .tar.gz archive."""
    try:
        with tarfile.open(tar_file_path, "w:gz") as tar:
            tar.add(directory_path, arcname=os.path.basename(directory_path))
        st.info(f"Compressed directory {directory_path} into {tar_file_path}.")
    except Exception as e:
        st.error(f"Error compressing directory: {e}")

# Decompress a .tar.gz archive into a directory
def decompress_tar_to_directory(tar_file_path, extract_to_path):
    """Decompresses a .tar.gz archive into a directory."""
    try:
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=extract_to_path)
        st.info(f"Decompressed {tar_file_path} to {extract_to_path}.")
    except Exception as e:
        st.error(f"Error decompressing tar file: {e}")

# Upload FAISS index to GCS
def upload_faiss_to_gcs(local_path, gcs_path):
    """Uploads a FAISS index (compressed) to Google Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Delete the existing index if it exists
        if blob.exists():
            blob.delete()
            st.info(f"Deleted existing FAISS index at {gcs_path}.")

        blob.upload_from_filename(local_path)
        st.success(f"FAISS index uploaded to {gcs_path}.")
    except Exception as e:
        st.error(f"Error uploading FAISS index to GCS: {e}")

# Download FAISS index from GCS
def download_faiss_from_gcs(gcs_path, local_path):
    """Downloads a FAISS index from Google Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        if blob.exists():
            blob.download_to_filename(local_path)
            st.info(f"Downloaded FAISS index from {gcs_path}.")
            return True
        else:
            st.warning(f"No FAISS index found at {gcs_path}.")
            return False
    except Exception as e:
        st.error(f"Error downloading FAISS index from GCS: {e}")
        return False

# Process tables into embeddings
def embed_extracted_tables(output_folder):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    table_chunks = []

    for file_name in os.listdir(output_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_folder, file_name)
            try:
                table_data = pd.read_json(file_path)
                table_text = table_data.to_string(index=False)
                table_chunks.append(table_text)
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
    
    return table_chunks

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Combine text and table chunks into FAISS vector store
def get_combined_vector_store(text_chunks, table_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_chunks = text_chunks + table_chunks
    local_dir = "faiss_index_combined"
    tar_file = "faiss_index_combined.tar.gz"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        # Create FAISS index
        vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
        vector_store.save_local(local_dir)

        # Compress the directory
        compress_directory_to_tar(local_dir, tar_file)

        # Upload to GCS
        upload_faiss_to_gcs(tar_file, gcs_path)

        # Clean up local files after uploading
        if os.path.exists(tar_file):
            os.remove(tar_file)
            st.info("Local compressed FAISS file deleted after uploading to GCS.")
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
            st.info("Local FAISS index directory deleted after uploading to GCS.")

    except Exception as e:
        st.error(f"Error creating or uploading FAISS vector store: {e}")

# Conversational QA chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    tar_file = "faiss_index_combined.tar.gz"
    local_dir = "faiss_index_combined"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        # Download FAISS index from GCS
        if download_faiss_from_gcs(gcs_path, tar_file):
            # Decompress the index
            decompress_tar_to_directory(tar_file, local_dir)

            # Load the FAISS index
            new_db = FAISS.load_local(local_dir, embeddings)

            # Search and respond
            docs = new_db.similarity_search(user_question, k=5)
            st.write("Retrieved Context:")
            for doc in docs:
                st.write(f"- {doc.page_content}")

            chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-1.5-pro"), chain_type="stuff")
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])

            # Clean up local files after usage
            if os.path.exists(tar_file):
                os.remove(tar_file)
                st.info("Local compressed FAISS file deleted after processing.")
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
                st.info("Local FAISS index directory deleted after processing.")
        else:
            st.error("FAISS index could not be loaded. Please process PDFs first.")
    except Exception as e:
        st.error(f"Error during query processing: {e}")

# Main Streamlit App
def main():
    st.set_page_config(page_title="FAISS Cloud Index", page_icon="💻", layout="wide")
    st.sidebar.title("Navigation")
    pages = ["Upload & Process PDFs", "Chat with PDFs"]
    selected_page = st.sidebar.radio("Choose a Page:", pages)

    if selected_page == "Upload & Process PDFs":
        st.title("Upload & Process PDFs")
        st.write("Upload your PDF documents for processing and embedding.")

        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # Extract text and tables
                        text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(text)

                        output_folder = "extracted_tables"
                        os.makedirs(output_folder, exist_ok=True)
                        extract_tables_from_pdfs("uploaded_pdfs", output_folder)
                        table_chunks = embed_extracted_tables(output_folder)

                        # Create FAISS index
                        get_combined_vector_store(text_chunks, table_chunks)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

    elif selected_page == "Chat with PDFs":
        st.title("Chat with PDFs")
        user_question = st.text_input("Enter your question:")
        if st.button("Submit Question"):
            if not user_question:
                st.warning("Please enter a question.")
            else:
                user_input(user_question)

if __name__ == "__main__":
    main()
