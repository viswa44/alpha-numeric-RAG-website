import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import pandas as pd
from google.cloud import storage
import tarfile
import shutil
from pydub import AudioSegment
import speech_recognition as sr
from extract_tables_json import extract_tables_from_pdfs
from work2 import embed_extracted_tables

#storage_client = storage.Client(project="rag4521")

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


# Audio transcription
def convert_audio_to_wav(audio_file):
    """Convert audio to WAV format."""
    try:
        audio = AudioSegment.from_file(audio_file)
        converted_file = "converted_audio.wav"
        audio.export(converted_file, format="wav")
        return converted_file
    except Exception as e:
        st.error(f"Error converting audio file: {e}")
        return None


def transcribe_audio(audio_file):
    """Transcribe audio file to text."""
    recognizer = sr.Recognizer()
    try:
        converted_file = convert_audio_to_wav(audio_file)
        if not converted_file:
            return None

        with sr.AudioFile(converted_file) as source:
            st.info("Processing the audio file for transcription...")
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            st.success(f"Transcribed Text: {text}")
            return text
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Error with Speech Recognition service: {e}")
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
    return None


# Streamlit pages
def render_chat():
    st.title("Chat with PDF using Gemini LLM")

    # PDF Upload
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
    if pdf_docs and st.button("Process PDFs"):
        text = extract_text_from_pdfs(pdf_docs)
        text_chunks = split_text_into_chunks(text)
        table_chunks = embed_extracted_tables("extracted_tables")
        create_and_upload_vector_store(text_chunks, table_chunks)

    # Text Query
    user_question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        process_user_query(user_question)

    # Audio Upload
    audio_file = st.file_uploader("Upload an audio file for transcription (e.g., WAV, MP3)", type=["wav", "mp3", "flac"])
    if audio_file and st.button("🎙️ Transcribe Audio and Query"):
        transcription = transcribe_audio(audio_file)
        if transcription:
            process_user_query(transcription)


def render_blog():
    st.title("Welcome to Demo Project")
    st.sidebar.title("Topics of project")
    sections = ["Introduction", "Requirements", "Challenges", "Contact Me"]
    choice = st.sidebar.radio("Go to", sections)

    if choice == "Introduction":
        st.header("Welcome to Demo Project!")
        st.write("""
        I'm excited to share my thoughts, insights, and challenges with you.
        Explore the posts listed in the sidebar to dive into specific topics.
        """)

    elif choice == "Requirements":
        st.header("Understanding Project")
        st.write("""
        ### Technical Components in the Project:
        1. Text and Table Processing
        2. Embedding and Search
        3. Conversational AI
        4. Voice Interaction

        ### Benefits:
        - Simplified Document Querying
        - Enhanced User Interaction
        - Semantic Understanding
        
        ### Conclusion:
        By integrating PDF processing, embeddings, and conversational AI, this project provides an efficient way to interact with documents.
        """)
    elif choice == "Challenges":
        st.header("Challenges with Poorly Structured Data")
        st.write("""
        Poorly structured data poses significant challenges in extracting meaningful relationships and insights, particularly in RAG applications.
        
        #### Common Challenges:
        - **Lack of Context**: Missing headers or inconsistent formatting can obscure relationships.
        - **Alpha-Numeric Confusion**: Identifiers like "Item123" or "Cat5" can be ambiguous without proper context.
        - **Entity Overlap**: Entities might overlap in meaning, making it hard to distinguish them.
        - **Data Sparsity**: Missing or incomplete data rows hinder analysis.
        - **Semantic Gap**: Natural language models struggle to interpret poorly organized data.

        **Addressing these challenges** requires robust pre-processing, advanced embeddings, and hybrid systems combining rule-based and AI methods.
        """)
    elif choice == "Contact Me":
        st.header("Contact Me")
        st.markdown("""
        Feel free to reach out:
        - **Email**: boyalaguntaviswa0009@gmail.com
        - **Mobile**: +91 6364539426
        - **LinkedIn**: [Viswatej's Profile](https://www.linkedin.com/in/viswatej-varma-55335b169/)
        """)


def main():
    st.sidebar.title("Navigation")
    pages = ["Blog", "Chat with PDFs"]
    selected_page = st.sidebar.radio("Choose a Page:", pages)

    if selected_page == "Blog":
        render_blog()
    elif selected_page == "Chat with PDFs":
        render_chat()


if __name__ == "__main__":
    main()
