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
from work2 import embed_extracted_tables

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Google Cloud Storage setup
storage_client = storage.Client()
bucket_name = "bucket-rag4521"  # Replace with your GCP bucket name


# Utility functions
def compress_directory(directory_path, tar_file_path):
    """Compress a directory into a tar.gz file."""
    try:
        with tarfile.open(tar_file_path, "w:gz") as tar:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=directory_path)
                    tar.add(full_path, arcname=arcname)
        st.info(f"Compressed {directory_path} into {tar_file_path}.")
    except Exception as e:
        st.error(f"Error compressing directory: {e}")


def decompress_tar(tar_file_path, extract_to_path):
    """Decompress a tar.gz file."""
    try:
        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=extract_to_path)
        st.info(f"Decompressed {tar_file_path} to {extract_to_path}.")
    except Exception as e:
        st.error(f"Error decompressing tar file: {e}")


def upload_to_gcs(local_path, gcs_path):
    """Upload a file to Google Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        # Delete existing object if exists
        if blob.exists():
            blob.delete()
            st.info(f"Deleted existing file at {gcs_path}.")

        blob.upload_from_filename(local_path)
        st.success(f"Uploaded file to {gcs_path}.")
    except Exception as e:
        st.error(f"Error uploading file to GCS: {e}")


def download_from_gcs(gcs_path, local_path):
    """Download a file from Google Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)

        if blob.exists():
            blob.download_to_filename(local_path)
            st.info(f"Downloaded file from {gcs_path}.")
            return True
        else:
            st.warning(f"No file found at {gcs_path}.")
            return False
    except Exception as e:
        st.error(f"Error downloading file from GCS: {e}")
        return False


# PDF processing
def extract_text_from_pdfs(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def split_text_into_chunks(text):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


# FAISS and embedding
def create_and_upload_vector_store(text_chunks, table_chunks):
    """Create FAISS vector store and upload to GCS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_chunks = text_chunks + table_chunks
    local_dir = "faiss_index_combined"
    tar_file = "faiss_index_combined.tar.gz"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        # Clean and prepare local directory
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir)

        # Create FAISS index
        vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
        vector_store.save_local(local_dir)

        # Verify files saved
        if not os.path.exists(f"{local_dir}/index.faiss"):
            st.error("FAISS index was not saved correctly.")
            return

        # Compress and upload to GCS
        compress_directory(local_dir, tar_file)
        upload_to_gcs(tar_file, gcs_path)

        # Clean up local files
        os.remove(tar_file)
        shutil.rmtree(local_dir)
    except Exception as e:
        st.error(f"Error creating or uploading FAISS vector store: {e}")


# Voice transcription
def transcribe_audio(audio_file=None):
    """Transcribe voice input from microphone or audio file."""
    recognizer = sr.Recognizer()
    try:
        if audio_file:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        else:
            with sr.Microphone() as source:
                st.info("Listening... Speak now.")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
        st.success(f"Transcribed text: {text}")
        return text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None


# Conversational QA
def process_user_query(user_question):
    """Process user query with FAISS and conversational AI."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    tar_file = "faiss_index_combined.tar.gz"
    local_dir = "faiss_index_combined"
    gcs_path = "faiss_index_combined.tar.gz"

    try:
        if download_from_gcs(gcs_path, tar_file):
            decompress_tar(tar_file, local_dir)

            if not os.path.exists(f"{local_dir}/index.faiss"):
                st.error("FAISS index is missing after decompression.")
                return

            vector_store = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_question, k=5)
            st.write("Retrieved Context:")
            for doc in docs:
                st.write(f"- {doc.page_content}")

            chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-1.5-pro"), chain_type="stuff")
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])

            os.remove(tar_file)
            shutil.rmtree(local_dir)
        else:
            st.error("Failed to load FAISS index. Please process PDFs first.")
    except Exception as e:
        st.error(f"Error during query processing: {e}")


# Streamlit UI
def render_about_page():
    """Render the About page."""
    st.title("Welcome to the Project!")
    st.write("""
        This project combines AI and Cloud technology for semantic document search and interaction.
        Features include PDF processing, FAISS-based vector search, and conversational AI.
    """)


def render_upload_page():
    """Render the PDF upload page."""
    st.title("Upload PDFs")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if pdf_docs:
            text = extract_text_from_pdfs(pdf_docs)
            text_chunks = split_text_into_chunks(text)
            table_chunks = embed_extracted_tables("extracted_tables")
            create_and_upload_vector_store(text_chunks, table_chunks)
        else:
            st.warning("Please upload at least one PDF.")


def render_chat_page():
    """Render the Chat page."""
    st.title("Chat with PDFs")
    user_question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        process_user_query(user_question) ######
    if st.button("🎙️ Speak"):
        voice_input = transcribe_audio()
        if voice_input:
            process_user_query(voice_input)


def main():
    """Main Streamlit app function."""
    st.set_page_config(page_title="FAISS Cloud App", page_icon="💻", layout="wide")
    st.sidebar.title("Navigation")
    pages = ["About the Project", "Upload & Process PDFs", "Chat with PDFs"]
    selected_page = st.sidebar.radio("Choose a Page:", pages)

    if selected_page == "About the Project":
        render_about_page()
    elif selected_page == "Upload & Process PDFs":
        render_upload_page()
    elif selected_page == "Chat with PDFs":
        render_chat_page()


if __name__ == "__main__":
    main()
