import streamlit as st
import pandas as pd
import spacy
import PyPDF2
import os
import joblib
import tempfile
from pdf2image import convert_from_bytes
from PIL import Image
from pathlib import Path
from io import BytesIO
from spacy_streamlit import visualize_ner
from labeler import SimpleLabeler

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

excel_path = 'master_label_entities.xlsx'
labeler = SimpleLabeler(excel_path)

def reading_pdf(uploaded_file, dpi=300) -> str:
    text_content = ''
    # Use BytesIO to create a file-like object from the uploaded file bytes
    file_stream = BytesIO(uploaded_file.getvalue())

    # Initialize the PDF reader with the file-like object
    pdf_reader = PyPDF2.PdfReader(file_stream)

    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            text_content += text.lower() + "\n"  # Append text of each page
    return text_content

def configure_retriever(uploaded_files, openai_api_key):
        # Read documents
        temp_dir = tempfile.TemporaryDirectory()
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Write the uploaded file to a temporary directory
                temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())
        
                # Determine the file type and use the appropriate loader
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(temp_filepath)
                elif uploaded_file.name.endswith(('.docx', '.doc')):
                    loader = Docx2txtLoader(temp_filepath)
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(temp_filepath)
                else:
                    continue  # Skip unsupported file types
                # Load the document content and add it to the documents list
                documents.extend(loader.load())

                # Split documents
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create embeddings and store in vectordb
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
        vectordb = FAISS.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

        return retriever

def display_greeting(msgs):
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

def display_clear_history_button(msgs):
    if st.button("Clear message history"):
        msgs.clear()

def display_chatbot_interface(uploaded_files, openai_api_key):

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    retriever = configure_retriever(uploaded_files, openai_api_key)

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    display_greeting(msgs)
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    message_container = st.container()
    with message_container:
        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

    user_query = st.chat_input(placeholder="Ask me anything!")
    if user_query:
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
    
    display_clear_history_button(msgs)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")
    pass

def display_pdf_interface(uploaded_file):
    processsed = False
    if uploaded_file is not None:
        with st.spinner('Processing...'):
            text_content = reading_pdf(uploaded_file)
            doc = labeler(text_content)  # Process text with SimpleLabeler
            colors = {
                'Development Progress': '#FF5733',      # A strong red for progress-related topics
                'Impact on Society': '#33FF57',         # A vibrant green for societal impacts
                'Healthcare': '#3357FF',                # A calming blue for healthcare
                'Education': '#FF33FF',                 # A pink for educational aspects
                'Urban AI Implementation': '#FFFF33',   # A bright yellow for urban AI implementations
                'Ethical Considerations': '#FF8333',    # An orange for ethics in AI
                'Public Policy': '#33FFF6',             # A teal for public policy discussions
                'Technological Advancements': '#F633FF',# A purple for technology advancements
                'AI and Employment': '#8C33FF',         # An indigo for AI's impact on employment
                'Transportation': '#33FF8A'             # A light green for transportation
            }
            # Visualization of entities
            labels = [label.upper() for label in labeler.labels]  # Use the labels from your labeler instance
            visualize_ner(doc, labels=labels, colors=colors)
            processed = True
    return processed

def app():
    st.set_page_config(page_title="PDF and Chatbot Interface", layout="wide")
    
    st.title("PDF Analysis Interface")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.")
        return

    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    if uploaded_files:
        uploaded_files = [uploaded_files] if not isinstance(uploaded_files, list) else uploaded_files
        # Splitting the page into two columns
        col1, col2 = st.columns([3, 3])  # Adjust the ratio as needed

        with col1:
            st.header("PDF Interface")
            processed = [display_pdf_interface(uploaded_file) for uploaded_file in uploaded_files]
        
        if all(processed):
            with col2:
                st.header("Chatbot")
                display_chatbot_interface(uploaded_files, openai_api_key)

if __name__ == "__main__":
    app()
