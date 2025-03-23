from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def initialize_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name="insight-ai-support-bot"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dimension=model.get_sentence_embedding_dimension()
    metric="cosine"
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name = index_name,
            dimension = dimension,
            metric = metric,
            spec = spec
        )

    index = pc.Index(index_name)
    return model, index

def help_center_scrape():
    urls_to_scrape = -1
    raw_data = []
    sitemap_url = "https://help.adjust.com/sitemap.xml"
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "xml")
    urls_and_lastmod = [(loc.text, loc.find_next("lastmod").text if loc.find_next("lastmod") else None) for loc in soup.find_all("loc")[:urls_to_scrape]]

    for url, lastmod in urls_and_lastmod:
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

def create_chunk_df(data, _model):
    chunk_size=256
    chunked_rows = []
    for i, row in data.iterrows():
        chunks = [row["content"][i:i + chunk_size] for i in range(0, len(row["content"]), chunk_size)]
        for i, chunk in enumerate(chunks):
            chunked_rows.append({
                "unique_id": f"{row['unique_id']}-chunk-{i}",
                "title": row["title"],
                "url": row["url"],
                "content": chunk
            })

    chunked_df = pd.DataFrame(chunked_rows)
    chunked_df["embedding"] = chunked_df.apply(create_embeddings, axis = 1, args=(_model,))
    return chunked_df

def create_embeddings(row, model):
    combined_text = f'''{row["title"]}{row["url"]}{row["content"]}'''
    return model.encode(combined_text, show_progress_bar = False)

def upsert_embeddings(data, index):
    batch_size = 100
    to_upsert = []

    for i, row in data.iterrows():
        vector_exists = index.fetch(row["unique_id"]).get("vectors")
        if not vector_exists:
            to_upsert.append((row["unique_id"], row["embedding"], {"title": row["title"], "url": row["url"], "text": row["content"]}))

    for i in range(0, len(to_upsert), batch_size):
        index.upsert(vectors=to_upsert[i : i + batch_size])

def build_index():
    # model, index = initialize_pinecone_index()
    # raw_data =  help_center_scrape()
    # data = create_chunk_df(raw_data, model)
    # upsert_embeddings(data, index)

    # return model, index

    return SentenceTransformer("all-MiniLM-L6-v2"), Pinecone(api_key=PINECONE_API_KEY).Index("insight-ai-support-bot")