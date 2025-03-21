# InsightAI

InsightAI is a Streamlit-based intelligent chatbot that provides accurate responses to queries about Adjust's product suite and AdTech industry by leveraging RAG (Retrieval Augmented Generation) technology.

**Live Demo**: This application has been deployed and is hosted via Streamlit Community at [https://adjust-insight-ai.streamlit.app/](https://adjust-insight-ai.streamlit.app/)

## Overview

InsightAI scrapes Adjust's help center documentation, chunks and embeds the content, and stores it in a Pinecone vector database. When users ask questions, the system retrieves the most relevant documentation and uses OpenAI's GPT model to generate contextually appropriate answers.

## Features

- **Documentation Scraping**: Automatically scrapes content from Adjust's help center
- **Vector Search**: Uses Pinecone for semantic search capabilities
- **Contextual Responses**: Provides answers based on relevant documentation
- **Source Citations**: Links to original documentation sources for verification
- **Streamlit Interface**: User-friendly chat interface for easy interaction

## Requirements

- Python 3.7+
- OpenAI API key
- Pinecone API key
- Streamlit

## Dependencies
openai
streamlit
pinecone-client
sentence-transformers
requests
beautifulsoup4
pandas