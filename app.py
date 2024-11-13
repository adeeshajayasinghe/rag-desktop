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
        Welcome to the DevOps Process Model Assistant! 
        This chatbot is here to guide you through every stage of the DevOps lifecycle — plan, code, build, test, deploy, operate, monitor, and improve. 
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
    st.markdown("- What is a blue-green deployment strategy?")
    st.markdown("""- What are the key stages of a DevOps pipeline?""")
    st.markdown("""- What are the best practices for security in DevOps?""")
    st.markdown("- What does 'shift-left testing' mean in a DevOps context?")
    st.markdown("""- How do you ensure code quality in a DevOps pipeline?""")
    st.markdown("- What is Infrastructure as Code?")


st.title("DevOps Model Assistant")
st.info("""Ask me questions related to DevOps practices!""")

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
