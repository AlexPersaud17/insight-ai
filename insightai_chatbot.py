from openai import OpenAI
import streamlit as st
from system_message import SYSTEM_MESSAGE


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
        stream=True
    )
    response_text = st.write_stream(response)
    return response_text


def write_sources(relevant_docs):
    st.write("Sources:")
    st.write("\n\n".join(set(
        f"[{url['metadata']['title']}]({url['metadata']['url']})" for url in relevant_docs)))


def run_chat(model, index, chat_container):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system",
             "content": SYSTEM_MESSAGE},
            {"role": "assistant",
             "content": "Hello! I'm InsightAI. How can I help you today?"}
        ]

    for message in st.session_state.messages[1:]:
        if message.get("display", True):
            with chat_container.chat_message(message["role"]):
                st.write(message["content"])

    if prompt := st.chat_input("How can I help?", max_chars=500):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container.chat_message("user"):
            st.write(prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_results = query_index(model, prompt, index)
                if query_results:
                    relevant_docs = match_scoring(query_results)
                    if relevant_docs:
                        formatted_context = format_context(relevant_docs)

                        st.session_state.messages.append({"role": "user",
                                                          "content": f"Context:\n{formatted_context}\n\nQuestion: {prompt}",
                                                          "display": False})

                        st.session_state.messages.append({"role": "assistant",
                                                          "content": llm_response(client)})
                        write_sources(relevant_docs)
                    else:
                        fallback_response = "I couldn't find relevant information. Please rephrase your question or visit Adjust's Help Center."
                        st.write(fallback_response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": fallback_response})
                else:
                    fallback_response = "I don't have information on that topic. Try asking about something else related to Adjust."
                    st.write(fallback_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": fallback_response})
