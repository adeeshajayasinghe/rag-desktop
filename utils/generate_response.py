from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


async def get_llm_response(query):
    # Get the directory containing the executable (assuming a single-file executable)
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Define a subdirectory within the executable's directory to store FAISS data
    vector_db_dir = os.path.join(script_dir, "vector_db")

    # load vector db
    vector_db = FAISS.load_local(
        vector_db_dir, huggingface_embeddings, allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=512,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. Act as a DevOps Process Model Assistant, focusing on providing clear, concise, and practical responses to help users with DevOps activities, including planning, coding, building, testing, deploying, operating, monitoring, feedback, and continuous improvement. Offer best practices, tool recommendations, and actionable steps for each stage, and integrate guidance on security and collaboration where relevant. Your responses should aim to help users optimize their DevOps workflows, enhance collaboration, and maintain security throughout the process. 
    2. If you don't know the answer, don't try to make up an answer. Just say "I can't find the answer within my knowledge base".
    3. Please provide your answer in paragraph form only, without using bullet points or numbered lists. 

    {context}

    Question: {question}

    Helpful Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return await retrievalQA.ainvoke({"query": query})
