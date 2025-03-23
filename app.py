from index_builder import build_index
from insightai_chatbot import run_chat
import streamlit as st

def page_init():
    st.set_page_config(page_title="InsightAI")
    st.title("InsightAI")

def app():
    model, index = build_index()
    run_chat(model, index)

if __name__ == "__main__":
    page_init()
    app()
