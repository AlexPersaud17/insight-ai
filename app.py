from index_builder import build_index
from insightai_chatbot import run_chat
from data_uploader import upload_new_data
import streamlit as st


def page_init():
    st.set_page_config(page_title="InsightAI")
    st.title("InsightAI")


def main():
    chat_tab, upload_tab = st.tabs(["Chat", "Upload Data"])
    model, index = build_index()
    with chat_tab:
        chat_container = st.container(height=600, border=False)
        run_chat(model, index, chat_container)
    with upload_tab:
        upload_new_data(model, index)


if __name__ == "__main__":
    page_init()
    main()
