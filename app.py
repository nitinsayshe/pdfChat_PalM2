import os
import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import GooglePalmEmbeddings


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    # return vector_store


def get_conversational_chain():
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = GooglePalmEmbeddings()
    new_db = FAISS.load_local("faiss_index",embeddings)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=new_db.as_retriever(),
        memory=memory)
    return conversation_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human : ", message.content)
        else:
            st.write("Bot : ", message.content)


def main():
    st.set_page_config("wizzo demo3")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    styl = f"""
      <style>
          .stTextInput {{
            position: fixed;
            bottom: 0rem;
            padding-bottom: 30px;
            padding-top: 20px;  
            background-color: #0E1117;
            z-index: 100;
          }}
      </style>
      """
    st.markdown(styl, unsafe_allow_html=True)
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain()
                st.success("Done")


if __name__ == "__main__":
    main()
