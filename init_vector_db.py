from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os
import shutil

def initialize_vector_db():
    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="llama3.2:3b")
    persist_directory = "rapipay_loan_db"
    
    # Clear existing DB if needed
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    # Load documents from docs folder
    loader = DirectoryLoader('docs/', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("No documents found in docs/ folder")
        return
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    texts = text_splitter.split_documents(documents)
    
    # Create and persist vector store
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector DB initialized with {len(texts)} chunks from {len(documents)} documents")

if __name__ == "__main__":
    print("Initializing Rapipay Loan Vector Database...")
    initialize_vector_db()
    print("Vector DB initialization complete!")