import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ==========================================
# CONFIG
# ==========================================

load_dotenv()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# READ PDF TEXT
# ==========================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:

                text += extracted_text

    return text


# ==========================================
# SPLIT TEXT INTO CHUNKS
# ==========================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ==========================================
# CREATE VECTOR STORE
# ==========================================

def get_vector_store(text_chunks):

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

        st.error(f"Error processing PDF: {str(e)}")

        return False


# ==========================================
# AUTO SELECT GEMINI MODEL
# ==========================================

def get_available_model():

    try:

        models = genai.list_models()

        available_models = []

        for m in models:

            if "generateContent" in m.supported_generation_methods:

                available_models.append(m.name)

        st.sidebar.write("Available Models:")
        st.sidebar.write(available_models)

        # Prefer flash model
        for model_name in available_models:

            if "flash" in model_name.lower():

                cleaned_name = model_name.replace("models/", "")

                st.sidebar.success(f"Using: {cleaned_name}")

                return cleaned_name

        # Fallback
        if available_models:

            cleaned_name = available_models[0].replace("models/", "")

            st.sidebar.success(f"Using: {cleaned_name}")

            return cleaned_name

        return None

    except Exception as e:

        st.error(f"Model detection failed: {str(e)}")

        return None


# ==========================================
# CREATE PROMPT + MODEL CHAIN
# ==========================================

def get_conversational_chain():

    selected_model = get_available_model()

    if not selected_model:

        return None

    prompt_template = """
    You are a helpful AI assistant.

    Answer the user's question ONLY from the provided context.

    If answer is not available in context,
    say:
    "Answer is not available in the provided PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    # Modern LCEL chain
    chain = prompt | model

    return chain


# ==========================================
# CLEAR CHAT HISTORY
# ==========================================

def clear_chat_history():

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Upload PDFs and ask questions."
        }
    ]


# ==========================================
# USER QUERY HANDLING
# ==========================================

def user_input(user_question):

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(
            user_question,
            k=4
        )

        if not docs:

            return {
                "output_text": "No relevant content found in PDFs."
            }

        # Build context manually
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        chain = get_conversational_chain()

        if chain is None:

            return {
                "output_text": "No Gemini model available."
            }

        response = chain.invoke(
            {
                "context": context,
                "question": user_question
            }
        )

        # Extract response safely
        if hasattr(response, "content"):

            output_text = response.content

        else:

            output_text = str(response)

        return {
            "output_text": output_text
        }

    except Exception as e:

        error_message = str(e)

        print(f"Error: {error_message}")

        if "429" in error_message:

            return {
                "output_text": "Gemini API quota exceeded. Try again later."
            }

        return {
            "output_text": f"Unexpected error: {error_message}"
        }


# ==========================================
# MAIN APP
# ==========================================

def main():

    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    st.title("Chat with PDFs using Gemini 🤖")

    st.write("Upload PDFs and ask questions.")

    # ==========================================
    # SIDEBAR
    # ==========================================

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

                    text_chunks = get_text_chunks(raw_text)

                    success = get_vector_store(text_chunks)

                    if success:

                        st.success("PDFs processed successfully!")

            else:

                st.error("Please upload at least one PDF.")

        st.button(
            "Clear Chat History",
            on_click=clear_chat_history
        )

    # ==========================================
    # CHAT HISTORY
    # ==========================================

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Upload PDFs and ask questions."
            }
        ]

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    # ==========================================
    # USER QUESTION
    # ==========================================

    user_question = st.chat_input(
        "Ask a question from PDFs"
    )

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


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    main()
