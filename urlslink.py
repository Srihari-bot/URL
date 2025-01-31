import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import fitz  # PyMuPDF
import os

# Set up page configuration
st.set_page_config(
    page_title="SBA INFO SOLUTION",
    page_icon="sba_info_solutions_logo (1).jpg",  # Path to your icon
    layout="wide",  # Wide layout
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLMs
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
google_llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Based on the provided context, please answer the following question in detail.
    Ensure that you cover all relevant aspects and provide a comprehensive response.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}  # Default to an empty dictionary if no metadata is provided

def extract_links_from_pdf(uploaded_file):
    links = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                # Extract hyperlinks
                if page.annots:
                    for annot in page.annots:
                        if 'uri' in annot:
                            links.append(annot['uri'])
    except Exception as e:
        st.error(f"Error extracting links from PDF: {e}")
    return links

def crawl_links(links):
    crawled_content = {}
    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                full_text = soup.get_text()
                crawled_content[link] = full_text
            else:
                crawled_content[link] = f"Failed to retrieve {link} (Status Code: {response.status_code})"
        except Exception as e:
            crawled_content[link] = f"Error crawling {link}: {e}"
    return crawled_content

def generate_summary(content):
    prompt = f"Please summarize the following content:\n\n{content}"
    response = google_llm.invoke(prompt)
    return response

def google_api_search(query):
    prompt = f"Search the web for the following query:\n\n{query}"
    response = google_llm.invoke(prompt)
    return response

def load_pdf_from_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def vector_embedding_from_texts(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    documents = []
    for text in texts[:20]:  # Limiting to the first 20 texts
        chunks = text_splitter.split_text(text)
        documents.extend([Document(page_content=chunk) for chunk in chunks])
    return FAISS.from_documents(documents, embeddings)

# Streamlit UI
st.title("SBA INFO SOLUTION")

# Upload PDF file
uploaded_file = st.file_uploader("Upload PDF files", type=["pdf"], key="pdf_uploader")

if uploaded_file is not None:
    links = extract_links_from_pdf(uploaded_file)
    
    if links:
        st.subheader("Extracted Links:")
        for link in links:
            st.write(link)
        
        if st.button("Crawl Extracted Links", key="crawl_links"):
            crawled_content = crawl_links(links)
            st.session_state.crawled_content = crawled_content
            st.session_state.links = links
            st.subheader("Crawled Content from Links:")
            for link, content in crawled_content.items():
                st.write(f"**Link:** {link}")
                st.write(f"**Content:** {content[:10000]}")
                st.write("---")
                
                if "crawled_texts" not in st.session_state:
                    st.session_state.crawled_texts = []
                st.session_state.crawled_texts.append(content)
                
                if st.button(f"Generate Summary for {link}", key=f"summary_{link}"):
                    summary = generate_summary(content)
                    st.write(f"**Summary for {link}:** {summary}")
                
                if st.button(f"Search Content for {link}", key=f"search_{link}"):
                    search_result = google_api_search(content[:10000])
                    st.write(f"**Search Result for {link}:** {search_result}")
    else:
        st.warning("No links found in the PDF.")

url = st.text_input("Enter a URL to crawl:", key="url_input")
question = st.text_input("Enter a Question:", key="question_input")

if st.button("Crawl URL", key="crawl_url"):
    if url:
        crawled_content = crawl_links([url])
        st.session_state.crawled_content = crawled_content
        st.session_state.url = url
        st.subheader("Crawled Content from URL:")
        for link, content in crawled_content.items():
            st.write(f"**Link:** {link}")
            st.write(f"**Content:** {content[:20000]}")
            st.write("---")
            
            if "crawled_texts" not in st.session_state:
                st.session_state.crawled_texts = []
            st.session_state.crawled_texts.append(content)
            
            if st.button(f"Generate Summary for {link}", key=f"url_summary_{link}"):
                summary = generate_summary(content)
                st.write(f"**Summary for {link}:** {summary}")
            
            if st.button(f"Search Content for {link}", key=f"url_search_{link}"):
                search_result = google_api_search(content[:20000])
                st.write(f"**Search Result for {link}:** {search_result}")
    else:
        st.warning("Please enter a valid URL.")

if st.button("Get Answer", key="get_answer"):
    if question and "crawled_texts" in st.session_state:
        vectors = vector_embedding_from_texts(st.session_state.crawled_texts)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Response time:", time.process_time() - start)
            st.write("Question:", question)
            st.write("Answer:", response['answer'])
        
        with col2:
            st.subheader("Extracted Content:")
            for link, content in st.session_state.crawled_content.items():
                st.write(f"**Link:** {link}")
                st.write(f"**Content:** {content[:10000]}")
                st.write("---")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("Please crawl URLs or upload PDFs first and enter a question.")

if "links" in st.session_state:
    st.subheader("Previously Extracted Links:")
    for link in st.session_state.links:
        st.write(link)

if "url" in st.session_state:
    st.subheader("Previously Crawled URL:")
    st.write(st.session_state.url)

if question:
    st.subheader("Entered Question:")
    st.write(question)
