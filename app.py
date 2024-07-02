import os
import asyncio
import streamlit as st
from utils.embed_data import embed_docs
from utils.generate_response import get_llm_response

UPLOAD_FOLDER = 'D:/projects/RAG/chatbot_api/src/data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the hospitals, patients,
        visits, physicians, and insurance payers in  a fake hospital system.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

    # Save uploaded PDFs locally
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save file to the specified directory
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"{uploaded_file.name} saved!")

            with st.spinner(f"Embedding {uploaded_file.name}..."):
                response = asyncio.run(embed_docs())
        
            if response:
                st.success(f"{uploaded_file.name} embedded successfully!")
            else:
                st.error(f"Failed to embed {uploaded_file.name}. Please try again.")

    st.header("Example Questions")
    st.markdown("- Which hospitals are in the hospital system?")
    st.markdown(
        """- What is the current wait time at wallace-hamilton hospital?"""
    )
    st.markdown(
        """- At which hospitals are patients complaining about billing and
        insurance issues?"""
    )
    st.markdown(
        "- What is the average duration in days for closed emergency visits?"
    )
    st.markdown(
        """- What are patients saying about the nursing staff at
        Castaneda-Hardy?"""
    )
    st.markdown(
        "- What was the total billing amount charged to each payer for 2023?"
    )
    st.markdown("- What is the average billing amount for medicaid visits?")
    st.markdown(
        "- Which physician has the lowest average visit duration in days?"
    )
    st.markdown("- How much was billed for patient 789's stay?")
    st.markdown(
        """- Which state had the largest percent increase in medicaid visits
        from 2022 to 2023?"""
    )
    st.markdown(
        "- What is the average billing amount per day for Aetna patients?"
    )
    st.markdown(
        """- How many reviews have been written from
                patients in Florida?"""
    )
    st.markdown(
        """- For visits that are not missing chief complaints,
       what percentage have reviews?"""
    )
    st.markdown(
        """- What is the percentage of visits that have reviews for
        each hospital?"""
    )
    st.markdown(
        """- Which physician has received the most reviews for this visits
        they've attended?"""
    )
    st.markdown("- What is the ID for physician James Cooper?")
    st.markdown(
        """- List every review for visits treated by physician 270.
        Don't leave any out."""
    )


st.title("Private Educational Chatbot")
st.info(
    """Ask me questions related to your uploaded documents!"""
)

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
                source = doc.metadata['source']
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
