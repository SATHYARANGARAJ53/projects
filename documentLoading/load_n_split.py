from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


FILE_PATH = "../documents/mcdr.pdf"

# create loader
loader = PyPDFLoader(FILE_PATH)
# split document
pages = loader.load_and_split()

#print(len(pages))

# embedding function
embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_func,
    persist_directory="../vector_db",
    collection_name="mcdr"
    )

# make vector store persistant
vectordb.persist()