import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAI

# Set up page configuration
st.set_page_config(
    page_title="SBA INFO SOLUTION",
    page_icon="sba_info_solutions_logo.jpg",  # Path to your icon
    layout="wide",  # Wide layout
)

# Google AI API key
api_key = "YOUR_GOOGLE_API_KEY"  # Replace with your actual API key
llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

def extract_links_from_pdf(uploaded_file):
    links = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                # Extract hyperlinks
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
                # Get the full text from the page
                full_text = soup.get_text()
                crawled_content[link] = full_text
            else:
                crawled_content[link] = f"Failed to retrieve {link} (Status Code: {response.status_code})"
        except Exception as e:
            crawled_content[link] = f"Error crawling {link}: {e}"
    return crawled_content

def generate_summary(content):
    # Use Google AI to generate a summary or process the content
    prompt = f"Please summarize the following content:\n\n{content}"
    response = llm.invoke(prompt)
    return response


# Streamlit UI
st.title("SBA INFO SOLUTION")



# Upload PDF file
uploaded_file = st.file_uploader("Upload PDF files", type=["pdf"])

if uploaded_file is not None:
    # Extract links from the PDF
    links = extract_links_from_pdf(uploaded_file)
    
    if links:
        st.subheader("Extracted Links:")
        for link in links:
            st.write(link)
        
        if st.button("Crawl Extracted Links"):
            crawled_content = crawl_links(links)
            st.subheader("Crawled Content from Links:")
            for link, content in crawled_content.items():
                st.write(f"**Link:** {link}")
                st.write(f"**Content:** {content[:10000]}")  # Display the first 10000 characters
                st.write("---")
                
                # Generate summary using Google AI
                if st.button(f"Generate Summary for {link}"):
                    summary = generate_summary(content)
                    st.write(f"**Summary for {link}:** {summary}")
    else:
        st.warning("No links found in the PDF.")

# URL entry for crawling
url = st.text_input("Enter a URL to crawl:")

if st.button("Crawl URL"):
    if url:
        crawled_content = crawl_links([url])
        st.subheader("Crawled Content from URL:")
        for link, content in crawled_content.items():
            st.write(f"**Link:** {link}")
            st.write(f"**Content:** {content[:20000]}")  # Display the first 20000 characters
            st.write("---")
            
            # Generate summary using Google AI
            if st.button(f"Generate Summary for {link}"):
                summary = generate_summary(content)
                st.write(f"**Summary for {link}:** {summary}")
    else:
        st.warning("Please enter a valid URL.")

