import streamlit as st
import index_builder as ib
from pypdf import PdfReader
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_file_data():
    option = st.radio("Select import method:", ["Upload PDF", "Scrape URL"])
    if option == "Upload PDF":
       return pdf_scraper()

    elif option == "Scrape URL":
        url_input = st.text_input("Enter the URL to scrape:")
        return url_scraper(url_input)

def pdf_scraper():
    title = st.text_input("Title of the document:")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_pdf:
        raw_data = []
        doc = PdfReader(uploaded_pdf)
        content = ""
        for page in doc.pages:
            content += page.extract_text()
        raw_data.append({
            "url": "",
            "title": title,
            "content": content,
            "unique_id": f"pdf-{title}-{time.time()}"
        })
    data = pd.DataFrame(raw_data)
    return data

def url_scraper(url, lastmod=datetime.today().strftime('%Y-%m-%d'), raw_data = []):
    if url:
        page_response = requests.get(url)
        page_soup = BeautifulSoup(page_response.content, "html.parser")
        title = page_soup.title.string if page_soup.title else "No title found"
        content = page_soup.get_text()
        unique_id = str(url) + "_" + (lastmod if lastmod else "no_lastmod")
        raw_data.append({
                "url": url,
                "title": title,
                "content": content,
                "unique_id": unique_id
            })

    data = pd.DataFrame(raw_data)
    return data

def help_center_scrape():
    urls_to_scrape = -1
    raw_data = []
    sitemap_url = "https://help.adjust.com/sitemap.xml"
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "xml")
    urls_and_lastmod = [(loc.text, loc.find_next("lastmod").text if loc.find_next("lastmod") else None) for loc in soup.find_all("loc")[:urls_to_scrape]]

    for url, lastmod in urls_and_lastmod:
        raw_data = url_scraper(url, lastmod, raw_data)

    return raw_data

def upload_new_data(model, index):
    raw_data = get_file_data()
    if st.button("Import"):
        st.write(f"Beginning import...")
        data = ib.create_chunk_df(raw_data, model)
        st.write(f"Chunking...")
        ib.upsert_embeddings(data, index)
        st.write(f"Upserting...")

        st.write(f"Import completed.")