from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def initialize_pinecone_index():
    pc = Pinecone()
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

    model, index = SentenceTransformer("all-MiniLM-L6-v2"), Pinecone().Index("insight-ai-support-bot")
    
    return model, index