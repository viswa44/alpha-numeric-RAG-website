import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import speech_recognition as sr  # For voice-to-text transcription
from extract_tables_json import extract_tables_from_pdfs
# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Function to check microphone (optional)
def transcribe_voice():
    """Converts spoken input to text using the SpeechRecognition library."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Voice recognition error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return None

# Function to process tables into embeddings
def embed_extracted_tables(output_folder):
    """Reads JSON files (extracted tables) from the output folder and creates embeddings."""
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

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    """Extracts text from each page of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """Splits the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

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

# Function for conversational QA chain
def user_input(user_question):
    """Handles the user's question by searching the combined FAISS index and generating a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index_combined", embeddings)
        docs = new_db.similarity_search(user_question, k=5)
        st.write("Retrieved Context:")
        for doc in docs:
            st.write(f"- {doc.page_content}")
        chain = load_qa_chain(ChatGoogleGenerativeAI(model="gemini-1.5-pro"), chain_type="stuff")
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error during query processing: {e}")

# Main function
def main():
    st.set_page_config(page_title="My Application", page_icon="💻", layout="wide")
    st.sidebar.title("Navigation")
    pages = ["Blog", "Chat with PDF"]
    selected_page = st.sidebar.radio("Choose a Page:", pages)

    if selected_page == "Blog":
        render_blog()
    elif selected_page == "Chat with PDF":
        render_chat()

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
    if st.button("🎙️ Speak"):
        spoken_question = transcribe_voice()
        if spoken_question:
            user_input(spoken_question)

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
