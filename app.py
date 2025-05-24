# Advanced PDF AI Assistant with Corrected Imports
import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain

st.set_page_config(page_title="📄 PDF AI Assistant", layout="wide")
st.title("📄 Advanced PDF AI Assistant")

@st.cache_data(show_spinner=False)
def process_documents(uploaded_files):
    all_chunks = []
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        all_docs.extend(documents)
        
        # Clean up temporary file
        os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = text_splitter.split_documents(all_docs)
    return all_docs, all_chunks

@st.cache_data(show_spinner=False)
def get_summary(_docs):
    llm_summary = Ollama(model="llama3.2")
    summary_chain = load_summarize_chain(llm_summary, chain_type="stuff")
    return summary_chain.run(_docs[:10])  # Limit for performance

@st.cache_resource
def create_vectorstore(_docs):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.from_documents(_docs, embeddings)

def create_qa_chain(vectorstore):
    llm_qa = Ollama(model="llama3.2")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question at the end.
    If you don't know the answer, just say you don't know, don't try to make up an answer.

    Context:
    {context}

    Question:
    {input}

    Helpful Answer:
    """)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm_qa, prompt)
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)
    
    return retrieval_chain

uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        all_docs, all_chunks = process_documents(uploaded_files)

    st.subheader("📝 Document Summary")
    try:
        summary = get_summary(all_chunks)
        st.write(summary)
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")

    with st.spinner("Creating vector store..."):
        vectorstore = create_vectorstore(all_chunks)

    qa_chain = create_qa_chain(vectorstore)

    st.subheader("💬 Ask Questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your PDFs:")

    if query:
        with st.spinner("Generating answer..."):
            try:
                result = qa_chain.invoke({"input": query})
                answer = result['answer']
                # Extract source documents if available
                sources = []
                if 'context' in result:
                    for doc in result['context']:
                        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                            sources.append(doc.metadata['source'])
                
                st.session_state.chat_history.append((query, answer, sources))
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {q}"):
                st.markdown(f"**Answer:** {a}")
                if sources:
                    st.markdown("**Sources:**")
                    for source in sources:
                        st.write(f"- {source}")

        # Download last answer
        last_answer = st.session_state.chat_history[-1][1]
        st.download_button("📥 Download Last Answer", last_answer, file_name="answer.txt")
