# Importing necessary libraries
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
import os

# Setting up environment variables for API keys and IDs
# Note: Replace YOUR_GOOGLE_API_KEY, YOUR_GOOGLE_CSE_ID, and YOUR_OPENAI_API_KEY with your actual keys
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # API key for Google Custom Search JSON API
# os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_CSE_ID" # Google Custom Search Engine ID
# os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1" # Base URL for OpenAI API
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # API key for OpenAI

# Configuring the Streamlit page
st.set_page_config(page_title="Technical Documentation AI Bot", page_icon="🌐")

# Function to setup and return the necessary objects for the bot
def settings():
    # Importing libraries and classes for embeddings and vector storage
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore
    
    # Setting up embeddings and vector store
    embeddings_model = OpenAIEmbeddings()  
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # Initializing the language model
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name='gpt-4')

    # Setting up the search API wrapper
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initializing the web retriever
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    return web_retriever, llm

# Class for handling stream updates
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)

# Class for handling the retrieval of documents
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)

# Setting up the Streamlit UI
st.header("`Technical Documentation AI Chat Bot`")
st.info("`I can answer technical questions in real time by checking documentation for AWS, GitHub, Fivetran, Looker, dbt, Prefect, & Snowflake.`")

# Initializing the retriever and language model
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input for the question
question = st.text_input("`Ask a question:`")

if question:
    # Setting up logging for the retriever
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    
    # Initializing the QA chain with the retriever
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Handling the answer generation and display
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    
    # Running the QA chain to get the answer
    result = qa_chain({"question": question}, callbacks=[retrieval_streamer_cb, stream_handler])
    
    # Displaying the answer and sources
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
