import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...") #Load
    loader = UnstructuredLoader(file_path="C:\Automation\langchain-course\mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting") #Split
    # Limit chunk size to 1000 characters, overlap = 0 means chunks will not have overlapping data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Embedding
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")#Store
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    print("finish")