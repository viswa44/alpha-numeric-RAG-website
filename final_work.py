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
from pydub import AudioSegment
import streamlit as st
import speech_recognition as sr

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)


# Google Cloud Storage setup
storage_client = storage.Client()
bucket_name = "bucket-rag4521"


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



# Function to split text into chunks
def get_text_chunks(text):#checked
    """Splits the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# FAISS and embedding
def create_and_upload_vector_store(text_chunks, table_chunks):#checked
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
def process_user_query(user_question):#checked
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

def split_text_into_chunks(text):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)
# PDF processing
def extract_text_from_pdfs(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
# Function to combine text and table chunks into FAISS vector store
def get_combined_vector_store(text_chunks, table_chunks):
    """Combines text and table chunks and creates a unified FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_chunks = text_chunks + table_chunks
    try:
        vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_combined")
        st.success("Combined FAISS vector store created and saved.")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
def get_pdf_text(pdf_docs): #checked
    """Extracts text from each page of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text
# Chat Page
def render_chat():
    st.title("Chat with PDF using Gemini LLM")
    st.write("""
    Upload your PDF documents and interact with them using advanced AI features like table extraction, embeddings, and conversational AI.
    """)

    with st.sidebar:
        st.write("### Upload your PDF Files:")
        pdf_docs = st.file_uploader(
            label="Choose PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            help="You can upload multiple PDF files for processing."
        )
        
        if pdf_docs:
            st.write("Uploaded Files:")
            for pdf in pdf_docs:
                st.write(f"📄 {pdf.name}")
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file before proceeding.")
                return
            
            with st.spinner("Processing..."):
                try:
                    input_folder = "uploaded_pdfs"
                    output_folder = "extracted_tables"
                    os.makedirs(input_folder, exist_ok=True)
                    os.makedirs(output_folder, exist_ok=True)

                    # Save uploaded PDFs to the input folder
                    for pdf in pdf_docs:
                        file_path = os.path.join(input_folder, pdf.name)
                        with open(file_path, "wb") as f:
                            f.write(pdf.read())
                        st.write(f"✅ Saved: {pdf.name}")

                    # Extract tables from the PDFs
                    st.info("Extracting tables from PDFs...")
                    extract_tables_from_pdfs(input_folder, output_folder)
                    st.success("Table extraction complete. JSON files created.")

                    # Embed extracted tables and process raw text
                    st.info("Embedding tables...")
                    table_chunks = embed_extracted_tables(output_folder)
                    
                    st.info("Extracting text from PDFs...")
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    # Combine text and table chunks into FAISS vector store
                    st.info("Combining text and table embeddings...")
                    get_combined_vector_store(text_chunks, table_chunks)

                    st.success("Processing complete! You can now ask questions.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # User interaction
    st.write("## Ask a Question from the PDF Files")
    user_question = st.text_input("Type your question below or click the microphone for voice input:")

    if st.button("Submit Question"):
        process_user_query(user_question)
    # if st.button("🎙️ Speak"):
    #     spoken_question = transcribe_voice()
    #     if spoken_question:
    #         user_input(spoken_question)
# Blog Page
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
    st.set_page_config(page_title="My Application", page_icon="💻", layout="wide")
    st.sidebar.title("Navigation")
    pages = ["Blog", "Chat with PDF"]
    selected_page = st.sidebar.radio("Choose a Page:", pages)

    if selected_page == "Blog":
        render_blog()
    elif selected_page == "Chat with PDF":
        render_chat()

if __name__ == "__main__":
    main()
