from openai import OpenAI
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import pandas as pd

def page_init():
    st.set_page_config(page_title="InsightAI")
    st.title("InsightAI")


@st.cache_resource(show_spinner=False)
def initialize_pinecone_index():
    pc = Pinecone(api_key = st.secrets["PINECONE_API_KEY"])
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

@st.cache_resource(show_spinner=False)
def help_center_scrape():
    count=0
    progress_bar = st.progress(count, "Scraping documentation...")
    urls_to_scrape = -1
    raw_data = []
    sitemap_url = "https://help.adjust.com/sitemap.xml"
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "xml")
    urls_and_lastmod = [(loc.text, loc.find_next("lastmod").text if loc.find_next("lastmod") else None) for loc in soup.find_all("loc")[:urls_to_scrape]]

    for url, lastmod in urls_and_lastmod:
        count += 1
        page_response = requests.get(url)
        page_soup = BeautifulSoup(page_response.content, "html.parser")
        title = page_soup.title.string if page_soup.title else "No title found"
        content = page_soup.get_text()
        # content = "\n".join(paragraph.text for paragraph in paragraphs[1:-1])
        unique_id = str(url) + "_" + (lastmod if lastmod else "no_lastmod")


        raw_data.append({
            "url": url,
            "title": title,
            "content": content,
            "unique_id": unique_id
        })
        progress_bar.progress(count/len(urls_and_lastmod), "Scraping documentation...")

    data = pd.DataFrame(raw_data)
    st.write("Scraping complete.")
    return data

@st.cache_resource(show_spinner=False)
def create_chunk_df(data, _model):
    progress_bar = st.progress(0, "Chunking and embedding...")
    chunk_size=256
    chunked_rows = []
    for i, row in data.iterrows():
        progress_bar.progress(i/len(data), "Chunking and embedding...")
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
    st.write("Chunking and embedding complete.")
    return chunked_df

def create_embeddings(row, model):
    combined_text = f'''{row["title"]}{row["url"]}{row["content"]}'''
    return model.encode(combined_text, show_progress_bar = False)

def upsert_embeddings(data, index):
    progress_bar = st.progress(0, "Initializing vectors...")
    batch_size = 100
    to_upsert = []

    for i, row in data.iterrows():
        vector_exists = index.fetch(row["unique_id"]).get("vectors")
        if not vector_exists:
            to_upsert.append((row["unique_id"], row["embedding"], {"title": row["title"], "url": row["url"], "text": row["content"]}))
        progress_bar.progress(i/len(data), "Initializing vectors...")
    progress_bar = st.progress(0, "Upserting vectors...")
    for i in range(0, len(to_upsert), batch_size):
        index.upsert(vectors=to_upsert[i : i + batch_size])
        progress_bar.progress(i/len(to_upsert), "Upserting vectors...")
    st.write("Upserting complete.")


def query_index(model, prompt, index):
    query_embedding = model.encode(prompt, show_progress_bar=False).tolist()
    query_results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    return query_results if query_results.matches and len(query_results.matches) > 0 else None

def match_scoring(query_results):
    score = 0.5
    return [{"score": match.score, "metadata": match.metadata} for match in query_results.matches if match.score > score]

def format_context(relevant_docs):
    formatted_context = ""
    for i, doc in enumerate(relevant_docs, 1):
        formatted_context += f"[Document {i}]\nTitle: {doc['metadata'].get('title', 'No title')}\n"
        formatted_context += f"Content: {doc['metadata'].get('text', 'No content')}\n\n"
    return formatted_context

def llm_response(client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages,
        temperature=0.0,
        max_tokens=5000,
        stream = True
    )
    response_text = st.write_stream(response) 
    return response_text

def chat(model, index):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": (
                '''
You are InsightAI, a highly knowledgeable assistant specializing in Adjust's product suite and the AdTech industry. Your role is to provide clear, accurate, and helpful responses to user inquiries based on the context and information available to you.

Contextual Understanding:
You should answer using ONLY the context provided. You do not have access to external sources or real-time information beyond the context given to you in the session.
If the question is beyond the scope of the provided information, inform the user politely that you dont have enough details to answer and encourage them to contact Adjust Support (support@adjust.com, unless the docs explicitly mention integrations@adjust.com) for further assistance.

Tone and Communication:
Be friendly, approachable, and professional.
Provide detailed responses, making sure to break down complex concepts for easy understanding.
Use clear examples and explanations when appropriate, ensuring the user feels confident and supported.

Adherence to Guidelines:
If a user asks something that is not covered in the provided context, acknowledge the gap in information and guide them towards Adjust's support.
Maintain accuracy and avoid speculating on information you do not have. If you dont know something, say so rather than providing incomplete or potentially incorrect answers.

User Support:
Your goal is to help users navigate the Adjust platform, solve their issues, and provide product-related insights.
Always aim to solve the users problem, if possible, based on the available context.
Stay patient and be respectful, even when the users question may seem unclear or too broad.

                '''
            )},
            {"role": "assistant", "content": "Hello! I'm InsightAI. How can I help you today?"}
        ]


    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            if "SYSTEMCONTEXT" not in message["content"]:
                st.write(message["content"])

    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_results = query_index(model, prompt, index)
                if query_results:
                    relevant_docs = match_scoring(query_results)
                    if relevant_docs:
                        formatted_context = format_context(relevant_docs)

                        st.session_state.messages.append({"role": "user",
                                                          "content": f"SYSTEMCONTEXTContext:\n{formatted_context}\n\nQuestion: {prompt}"})

                        response_text = llm_response(client)

                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        st.write("Sources:")
                        st.write("\n\n".join(set(f"[{url['metadata']['title']}]({url['metadata']['url']})" for url in relevant_docs)))
                    else:
                        fallback_response = "I couldn't find relevant information. Please rephrase your question or visit Adjust's Help Center."
                        st.write(fallback_response)
                        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
                else:
                    fallback_response = "I don't have information on that topic. Try asking about something else related to Adjust."
                    st.write(fallback_response)
                    st.session_state.messages.append({"role": "assistant", "content": fallback_response})


def main():
    model, index = initialize_pinecone_index()
    if "data" not in st.session_state:
        st.session_state.data = help_center_scrape()

    if "chunked_data" not in st.session_state:
        st.session_state.chunked_data = create_chunk_df(st.session_state.data, model)

    if "embeddings_upserted" not in st.session_state:
        upsert_embeddings(st.session_state.chunked_data, index)
        st.session_state.embeddings_upserted = True

    chat(model, index)

if __name__ == "__main__":
    main()



