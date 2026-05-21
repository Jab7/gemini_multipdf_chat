import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from google.generativeai.types import (
    BlockedPromptException,
    StopCandidateException,
    BrokenResponseError,
    IncompleteIterationError,
)

# =========================
# CONFIG
# =========================

load_dotenv()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

# =========================
# PDF TEXT EXTRACTION
# =========================

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

    return text


# =========================
# TEXT CHUNKING
# =========================

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_text(text)

    return chunks


# =========================
# VECTOR STORE
# =========================

def get_vector_store(text_chunks):

    if not text_chunks:
        st.error("No text chunks found.")
        return False

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_texts(
            text_chunks,
            embedding=embeddings
        )

        vector_store.save_local("faiss_index")

        return True

    except Exception as e:
        st.error(f"Error processing the PDF: {str(e)}")
        print(f"Embedding Error: {e}")

        return False


# =========================
# CONVERSATIONAL CHAIN
# =========================

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context.

    If the answer is not available in the provided context,
    just say:

    "answer is not available in the context"

    Do not provide incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# =========================
# CLEAR CHAT
# =========================

def clear_chat_history():

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload PDFs and ask me questions."
        }
    ]


# =========================
# USER INPUT
# =========================

def user_input(user_question):

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()

        response = chain(
            {
                "input_documents": docs,
                "question": user_question
            },
            return_only_outputs=True
        )

        print(response)

        return response

    except BlockedPromptException as e:

        print(f"Prompt blocked: {e}")

        return {
            "output_text": "Request blocked by Gemini safety filters."
        }

    except StopCandidateException as e:

        print(f"Response stopped: {e}")

        return {
            "output_text": "Response stopped due to safety concerns."
        }

    except (BrokenResponseError, IncompleteIterationError) as e:

        print(f"Response error: {e}")

        return {
            "output_text": "Error while generating response."
        }

    except Exception as e:

        print(f"Unexpected error: {e}")

        return {
            "output_text": f"Unexpected error: {str(e)}"
        }


# =========================
# MAIN APP
# =========================

def main():

    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    st.title("Chat with PDF using Gemini 🤖")

    st.write("Upload PDFs and ask questions.")

    # =========================
    # SIDEBAR
    # =========================

    with st.sidebar:

        st.title("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if pdf_docs:

                with st.spinner("Processing PDFs..."):

                    raw_text = get_pdf_text(pdf_docs)

                    mtext_chunks = get_text_chunks(raw_text)

                    success = get_vector_store(text_chunks)

                    if success:
                        st.success("PDFs processed successfully!")

            else:
                st.error("Please upload at least one PDF.")

        st.button(
            "Clear Chat History",
            on_click=clear_chat_history
        )

    # =========================
    # CHAT HISTORY
    # =========================

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload PDFs and ask me questions."
            }
        ]

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    # =========================
    # USER PROMPT
    # =========================

    user_question = st.chat_input("Ask a question from PDFs")

    if user_question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.write(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = user_input(user_question)

                output_text = response.get(
                    "output_text",
                    "No response generated."
                )

                st.write(output_text)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": output_text
                    }
                )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    main()
