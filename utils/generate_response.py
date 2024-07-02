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
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

async def get_llm_response(query):
    # load vector db
    vector_db=FAISS.load_local("D:/projects/RAG/chatbot_api/src/vector_db", huggingface_embeddings, allow_dangerous_deserialization=True)

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way as paragraphs.

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
        chain_type_kwargs={"prompt": PROMPT}
    )

    return await retrievalQA.ainvoke({"query": query})