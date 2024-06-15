


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class Loader:
    def __init__(self):
        self.persist_directory = os.getenv('PERSISTANT_STORAGE_DIR')
        self.data_directory = os.getenv('DATA_DIR')
        self.embedding = OpenAIEmbeddings()
    def get_retriever(self):
        if os.path.exists(self.persist_directory):
            choice = input(f"Directory {self.persist_directory} already exists. Do you want to update it? (y/n)")
            if choice.lower() == 'n':
                vectordb = Chroma(persist_directory=self.persist_directory, 
                  embedding_function=self.embedding)


                retriever = vectordb.as_retriever()
                return retriever

        loader = DirectoryLoader(self.data_directory+'/', glob="./*.pdf", loader_cls=PyPDFLoader)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
            
        vectordb = Chroma.from_documents(documents=texts, 
                                        embedding=self.embedding,
                                        persist_directory=self.persist_directory)


        retriever = vectordb.as_retriever()
        return retriever
        
        



# # Docs to index
# urls = [
#     
# ]

# # Load
# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# # Split
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)

# # Add to vectorstore
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embd,
# )
