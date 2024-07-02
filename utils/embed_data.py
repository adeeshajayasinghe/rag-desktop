from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


async def embed_docs():
    # Load pdf files in the local directory
    loader = PyPDFDirectoryLoader("D:/projects/RAG/chatbot_api/src/data")

    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap  = 50,
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={'device':'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load existing vector store
    vectorstore = FAISS.load_local("D:/projects/RAG/chatbot_api/src/vector_db", huggingface_embeddings, allow_dangerous_deserialization=True)

    # Add new embeddings
    await vectorstore.aadd_documents(docs_after_split)

    # Save vector database
    vectorstore.save_local("D:/projects/RAG/chatbot_api/src/vector_db")

    return