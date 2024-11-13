import os
import asyncio
import streamlit as st
from utils.embed_data import embed_docs
from utils.generate_response import get_llm_response

# Get the directory containing the executable (assuming a single-file executable)
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define a subdirectory within the executable's directory to store docs
data_dir = os.path.join(script_dir, "utils/data")

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions relevant to your uploaded content.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload your PDFs", type=["pdf"], accept_multiple_files=True
    )

    # Save uploaded PDFs locally
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save file to the specified directory
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"{uploaded_file.name} saved!")

            with st.spinner(f"Embedding {uploaded_file.name}..."):
                response = asyncio.run(embed_docs())

            # st.success(f"{uploaded_file.name} embedded successfully!")

            if response:
                st.success(f"{uploaded_file.name} embedded successfully!")
            else:
                st.error(f"Failed to embed {uploaded_file.name}. Please try again.")

    st.header("Example Questions")
    st.markdown("- What is RAG?")
    st.markdown(
        """- How my knowledge base ensures it remains relevant and up-to-date?"""
    )
    st.markdown("""- How can I fine-tune the chatbot's knowledge base?""")
    st.markdown(
        "- Can I use this chatbot to create and maintain my own personalised study companion?"
    )
    st.markdown(
        """- How does the chatbot learn from the specific information in the uploaded documents?"""
    )
    st.markdown(
        "- What steps are involved in the practical implementation of this document-based chatbot?"
    )


st.title("Private Educational Chatbot")
st.info("""Ask me questions related to your uploaded documents!""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "result" in message.keys():
            st.markdown(message["result"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "result": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = asyncio.run(get_llm_response(prompt))
        print("response: ", response)
        if response:
            output_text = response["result"]
            source_docs = response["source_documents"]

            unique_sources = set()
            for doc in source_docs:
                source = doc.metadata["source"]
                if source:
                    unique_sources.add(source)
            explanation = "Sources:\n" + "\n".join(unique_sources)

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "result": output_text,
            "explanation": explanation,
        }
    )
