from openai import OpenAI
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

nltk.download('stopwords')
nltk.download('punkt')

st.set_page_config(page_title="InsightAI")
st.title("InsightAI")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Creating PC Index
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

model, index = initialize_pinecone_index()

# Scrape HC data
def help_center_scrape():
    urls_to_scrape = 100
    raw_data = []
    sitemap_url = "https://help.adjust.com/sitemap.xml"
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, "xml")
    urls_and_lastmod = [(loc.text, loc.find_next("lastmod").text if loc.find_next("lastmod") else None) for loc in soup.find_all("loc")[:urls_to_scrape]]


    for url, lastmod in urls_and_lastmod:
        page_response = requests.get(url)
        page_soup = BeautifulSoup(page_response.content, "html.parser")
        title = page_soup.title.string if page_soup.title else "No title found"
        paragraphs = page_soup.find_all("p")
        content = "\n".join(paragraph.text for paragraph in paragraphs[1:-1])
        unique_id = str(url) + "_" + (lastmod if lastmod else "no_lastmod")

        raw_data.append({
            "url": url,
            "title": title,
            "content": content,
            "unique_id": unique_id
        })

    data = pd.DataFrame(raw_data)
    return data

data = help_center_scrape()

# Text pre-processing

def text_preprocessing(data):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    data['processed_content']=data['content'].str.lower()
    data['processed_content']=data['processed_content'].apply(
        lambda x: ' '.join(
            [word for word in x.split() if word not in (stop_words)]
        )
    )
    data['processed_content']=data["processed_content"].apply(
        lambda x: re.sub(r"[^\w\s]", "", x)
    )
    data['tokenized'] = data['processed_content'].apply(
        lambda x: word_tokenize(x)
    )
    data['lemmatized']=data['tokenized'].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )
    return data
data = text_preprocessing(data)

# Creating embeddings and upsert to VS
def upsert_embeddings(data):
    existing_vectors = index.fetch(ids=[str(i) for i in range(len(data))])
    existing_ids = {vector["id"] for vector in existing_vectors.vectors}

    batch_size = 100
    to_upsert = []

    for i, row in data.iterrows():
        if row["unique_id"] not in existing_ids:
            embedding = model.encode(row["processed_content"], show_progress_bar=False).tolist()
            to_upsert.append((row["unique_id"], embedding, {"title": row["title"], "url": row["url"], "text": row["content"][:1000]}))

    for i in range(0, len(to_upsert), batch_size):
        index.upsert(vectors=to_upsert[i : i + batch_size])

upsert_embeddings(data)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pinecone_retriever = PineconeVectorStore(index=index, embedding=embedding)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=st.secrets["OPENAI_API_KEY"])

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=pinecone_retriever.as_retriever(),
    memory=memory
)

if "messages" not in st.session_state:
    st.session_state.messages = [
         {"role": "system", "content": (
            "You are InsightAI, a helpful assistant for Adjust's product. "
            "Use ONLY the context provided to answer questions. "
            "If the answer is not in the context, say you don't have enough information "
            "and suggest they contact Adjust support for more details. "
            "Be friendly and very detailed."
        )},
        {"role": "assistant", "content": "Hello! I'm InsightAI. How can I help you today?"}
    ]
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask me anything about Adjust..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": prompt, "chat_history": st.session_state.messages})
            response_text = response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.write(response_text)

            query_embedding = model.encode(prompt, show_progress_bar=False).tolist()
            query_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            print("Query Results:", query_results)

            # query_embedding = model.encode(prompt, show_progress_bar=False).tolist()
            # query_results = index.query(
            #     vector = [query_embedding],
            #     top_k = 5,
            #     include_metadata=True
            # )
            # if query_results.matches and len(query_results.matches) > 0:
            #     relevant_docs = [
            #         {"score": match.score, "metadata": match.metadata} for match in query_results.matches if match.score > 0.1
            #     ]
            #     if relevant_docs:
            #         formatted_context = ""
            #         for i, doc in enumerate(relevant_docs, 1):
            #             formatted_context += f"[Document {i}]\nTitle: {doc['metadata'].get('title', 'No title')}\n"
            #             formatted_context += f"Content: {doc['metadata'].get('content', 'No content')}\n\n"
            #         messages = [
            #             {"role": "system", "content": (
            #                 "You are InsightAI, a helpful assistant for Adjust's product. "
            #                 "Use ONLY the context provided to answer questions. "
            #                 "If the answer is not in the context, say you don't have enough information "
            #                 "and suggest they contact Adjust support for more details. "
            #                 "Be friendly and very detailed."
            #             )},
            #             {"role": "user", "content": f"Context:\n{formatted_context}\n\nQuestion: {prompt}"}
            #         ]
            #         response = client.chat.completions.create(
            #             model="gpt-4o-mini",
            #             messages=messages,
            #             temperature=0.7,
            #             max_tokens=500,
            #             stream = True
            #         )
            #         response_text = st.write_stream(response)
            #         st.session_state.messages.append({"role": "assistant", "content": response_text})
            #     else:
            #         fallback_response = "I couldn't find relevant information. Please rephrase your question or visit Adjust's Help Center."
            #         st.write(fallback_response)
            #         st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            # else:
            #     fallback_response = "I don't have information on that topic. Try asking about something else related to Adjust."
            #     st.write(fallback_response)
            #     st.session_state.messages.append({"role": "assistant", "content": fallback_response})