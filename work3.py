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
from extract_tables_json import extract_tables_from_pdfs
from work2 import embed_extracted_tables, get_pdf_text, get_text_chunks

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Set up Google Cloud Storage client
storage_client = storage.Client()
bucket_name = "bucket-rag4521"  # Replace with your GCS bucket name

# Compress a directory into a .tar.gz archive
def compress_directory_to_tar(directory_path, tar_file_path):
    """Compresses a directory into a .tar.gz archive."""
    try:
        with tarfile.open(tar_file_path, "w:gz") as tar:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=directory_path)  # Relative path
                    tar.add(full_path, arcname=arcname)
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

# Combine text and table chunks into FAISS vector store
def get_combined_vector_store(text_chunks, table_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_chunks = text_chunks + table_chunks
    local_dir = "faiss_index_combined"
    tar_file = "faiss_index_combined.tar.gz"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)

        os.makedirs(local_dir)

        # Create FAISS index
        vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
        vector_store.save_local(local_dir)

        # Verify saved files
        st.write("FAISS files after saving:", os.listdir(local_dir))
        if not os.path.exists(f"{local_dir}/index.faiss"):
            st.error("FAISS index was not saved correctly.")
            return

        # Compress the directory
        compress_directory_to_tar(local_dir, tar_file)

        # Upload to GCS
        upload_faiss_to_gcs(tar_file, gcs_path)

        # Cleanup
        os.remove(tar_file)
        shutil.rmtree(local_dir)
        st.info("Local files cleaned up after upload.")
    except Exception as e:
        st.error(f"Error creating or uploading FAISS vector store: {e}")

# Conversational QA chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    tar_file = "faiss_index_combined.tar.gz"
    local_dir = "faiss_index_combined"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        # Download and decompress FAISS index
        if download_faiss_from_gcs(gcs_path, tar_file):
            decompress_tar_to_directory(tar_file, local_dir)

            # Verify files after decompression
            st.write("FAISS files after decompression:", os.listdir(local_dir))
            if not os.path.exists(f"{local_dir}/index.faiss"):
                st.error("FAISS index files are missing after decompression.")
                return

            # Load FAISS index
            new_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)

            # Perform similarity search
            docs = new_db.similarity_search(user_question, k=5)
            st.write("Retrieved Context:")
            for doc in docs:
                st.write(f"- {doc.page_content}")

            # QA Chain
            chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-1.5-pro"), chain_type="stuff")
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])

            # Cleanup
            os.remove(tar_file)
            shutil.rmtree(local_dir)
            st.info("Local files cleaned up after query.")
        else:
            st.error("FAISS index could not be downloaded. Please process PDFs first.")
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
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text)
                    table_chunks = embed_extracted_tables("extracted_tables")
                    get_combined_vector_store(text_chunks, table_chunks)
            else:
                st.warning("Please upload a PDF.")

    elif selected_page == "Chat with PDFs":
        st.title("Chat with PDFs")
        user_question = st.text_input("Enter your question:")
        if st.button("Submit Question"):
            user_input(user_question)

if __name__ == "__main__":
    main()
