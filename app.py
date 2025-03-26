from index_builder import build_index
from insightai_chatbot import run_chat
from data_uploader import upload_new_data, print_upload_history
import streamlit as st


def page_init():
    st.set_page_config(page_title="InsightAI")

    if not st.experimental_user.is_logged_in:
        st.button("Login with Google", on_click=st.login)
        st.stop()

    st.button("Log out", on_click=st.logout)
    st.title("InsightAI")


def main():
    chat_tab, upload_tab, upload_history = st.tabs(
        ["Chat", "Upload Data", "Upload History"])
    model, index = build_index()
    with chat_tab:
        chat_container = st.container(height=600, border=False)
        run_chat(model, index, chat_container)
    with upload_tab:
        upload_new_data(model, index)
    with upload_history:
        print_upload_history()


if __name__ == "__main__":
    page_init()
    main()
