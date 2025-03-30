""" 
app.py

Streamlit app for uploading and analyzing scientific papers using RAG. 
Allows users to upload a PDF, extract text, store it in a vector store, 
and interact with the document via a chatbot.
"""

import streamlit as st

from rag.document_processing import DocumentProcessor
from rag.llm import LLMQuery

model = LLMQuery()
processor = DocumentProcessor()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "upload"


def go_to_chat():
    st.session_state.page = "chat"


# Upload Page
if st.session_state.page == "upload":
    st.title("📄 AI-Powered Scientific Paper Chatbot")
    st.write("Upload a scientific paper and start asking questions!")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

        with st.spinner(text="In progress...", show_time=True):
            text = processor.read_pdf(uploaded_file)
            documents = processor.create_document(text.split("\n"))
            processor.load_documents(documents)

        st.success("File uploaded successfully!")
        st.button("Proceed to Chat", on_click=go_to_chat)

# Chat Page
elif st.session_state.page == "chat":
    st.title("💬 Chat with Your Paper")
    st.write("Ask me anything about the uploaded document!")

    # Retrieve uploaded file
    uploaded_file = st.session_state.get("uploaded_file", None)

    if uploaded_file:
        st.write(f"📄 **Uploaded File:** {uploaded_file.name}")
    else:
        st.warning("No file uploaded. Please go back to the upload page.")
        if st.button("Go Back"):
            st.session_state.page = "upload"
            st.experimental_rerun()  # Refresh the page

    # Chat input
    user_question = st.text_input("Ask a question:")

    if user_question:
        relevant_docs = processor.retriever.invoke(user_question)

        qa_chain, llm = model.create_qa_chain()
        response = model.send_query(user_question, relevant_docs, qa_chain)

        st.write(response)

        st.write(relevant_docs)
