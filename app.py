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
        run_chat(model, index)
    with upload_tab:
        upload_new_data(model, index)


if __name__ == "__main__":
    page_init()
    main()
