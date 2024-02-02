import chromadb
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import RetrievalQA

def search_from_vectordb(query):
        # 在 prompt 中添加相关字段
        collection = "question"
        db_path = "./real_agents/data_agent/db/" + collection
        persistent_client = chromadb.PersistentClient(path=db_path)
        # embedding_function = SentenceTransformerEmbeddings(
        #     model_name="all-MiniLM-L6-v2")
        embedding_function = OpenAIEmbeddings()
        langchain_chroma = Chroma(
            client=persistent_client,
            collection_name=collection,
            embedding_function=embedding_function,
        )
        docs = langchain_chroma.similarity_search(query, k=8)
        doc_str = "".join(elem.page_content + "\n" for elem in docs)
        return doc_str
