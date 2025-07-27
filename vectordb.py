import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import logging

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3:4b")
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model=EMBEDDING_MODEL)
persist_directory = "rapipay_loan_db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_vector_db():
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
        loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Initialized vector DB with {len(texts)} chunks")
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb

vectordb = initialize_vector_db()
