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


@st.cache_resource
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

@st.cache_resource
def help_center_scrape():
    urls_to_scrape = -1
    raw_data = []
    sitemap_url = "https://help.adjust.com/sitemap.xml"
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "xml")
    urls_and_lastmod = [(loc.text, loc.find_next("lastmod").text if loc.find_next("lastmod") else None) for loc in soup.find_all("loc")[:urls_to_scrape]]

    # st.write("Data scraped:")
    for url, lastmod in urls_and_lastmod:
        page_response = requests.get(url)
        page_soup = BeautifulSoup(page_response.content, "html.parser")
        title = page_soup.title.string if page_soup.title else "No title found"
        paragraphs = page_soup.find_all("p")
        content = "\n".join(paragraph.text for paragraph in paragraphs[1:-1])
        unique_id = str(url) + "_" + (lastmod if lastmod else "no_lastmod")

        # st.write(f"[{title}]({url})") 

        raw_data.append({
            "url": url,
            "title": title,
            "content": content,
            "unique_id": unique_id
        })

    data = pd.DataFrame(raw_data)
    st.write("Scraping complete.") 
    return data

@st.cache_resource
def create_chunk_df(data, _model):

    def chunk_text(tokens):
        chunk_size=256
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    chunked_rows = []
    for _, row in data.iterrows():
        chunks = chunk_text(row["content"])
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
    print("\n\n\nHERE1n\n\n")
    # existing_vectors = index.fetch(ids=[str(i) for i in range(len(data))])
    # existing_ids = {vector["id"] for vector in existing_vectors.vectors}

    batch_size = 1000
    to_upsert = []

    for i, row in data.iterrows():
        print("\n\n\nHERE2\n\n\n")
        # if row["unique_id"] not in existing_ids:
        to_upsert.append((row["unique_id"], row["embedding"], {"title": row["title"], "url": row["url"], "text": row["content"]}))
    for i in range(0, len(to_upsert), batch_size):
        print("\n\n\nHERE3\n\n\n")
        print(to_upsert[i : i + batch_size])
        index.upsert(vectors=to_upsert[i : i + batch_size])
    st.write("Upserting complete.")


def chat(model, index, client):
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
            st.write(message["content"])

    if prompt := st.chat_input("How can I help?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_embedding = model.encode(prompt, show_progress_bar=False).tolist()
                query_results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                print("Query Results:", query_results)

                if query_results.matches and len(query_results.matches) > 0:
                    relevant_docs = [
                        {"score": match.score, "metadata": match.metadata} for match in query_results.matches if match.score > 0.5
                    ]
                    print("Relevant Docs:", relevant_docs)
                    if relevant_docs:
                        formatted_context = ""
                        for i, doc in enumerate(relevant_docs, 1):
                            formatted_context += f"[Document {i}]\nTitle: {doc['metadata'].get('title', 'No title')}\n"
                            formatted_context += f"Content: {doc['metadata'].get('text', 'No content')}\n\n"

                        st.session_state.messages.append({"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {prompt}"})
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=st.session_state.messages,
                            temperature=0.0,
                            max_tokens=500,
                            stream = True
                        )
                        response_text = st.write_stream(response)
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
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model, index = initialize_pinecone_index()
    data = help_center_scrape()
    data = create_chunk_df(data, model)
    upsert_embeddings(data, index)
    chat(model, index, client)

if __name__ == "__main__":
    main()



