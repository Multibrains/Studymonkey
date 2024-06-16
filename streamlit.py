from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import PyPDF2
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import openai

retriever_type = "SIMILARITY SEARCH"
api_key = os.environ.get('OPENAI_API_KEY')
embedding_option = "Open AI Embedding"


st.set_page_config(
    page_title="StudyMonkey",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)




st.title(":red[StudyMonkey] — Learn Better. 📚", anchor=False)
st.write("""
With StudyMonkey, you can effortlessly convert PDFs into interactive chat sessions.
Simply upload your PDF documents, and the app will transform the content into a chat format, making studying more engaging and dynamic.

**How does it work?**
1. Upload a document (.txt, .pdf, or .md)
2. Ask questions!
""")




@st.cache_data
def load_docs(files):
    # st.info("`Reading doc ...`")
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    # st.info("`Splitting doc ...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()

    return splits

if uploaded_files:

    # Doc Reader Button

    loaded_text = load_docs(uploaded_files)
    # st.write("Documents uploaded and processed.")


    # Split Text /w RecursiveTextSplitter

    splitter_type = "RecursiveTextSplitter"
    splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)
    
    # Embed using OpenAI embeddings
    # Embed using OpenAI embeddings or HuggingFace embeddings
    
    embeddings = OpenAIEmbeddings()
        

    retriever = create_retriever(embeddings, splits, retriever_type)

    # Initialize the RetrievalQA chain with streaming output
    callback_handler = StreamingStdOutCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat_openai = ChatOpenAI(streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

    st.session_state.user_question = st.text_input("Enter your question:")
    with st.spinner("Thinking... 🤔"):
        if st.session_state.user_question:
                answer = qa.run(st.session_state.user_question)
                st.write("**Answer:**", answer)

   
