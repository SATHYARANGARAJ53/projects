from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from decouple import config


TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="This is the source documents there are 4 main documents,  `document 1`, `document 2`, `document 3`, `document 4`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the details of Python",
        type="integer",
    ),
]

document_content_description = "Info on Python Programming Language"
llm = OpenAI(temperature=0, openai_api_key=config("OPENAI_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)


docs = retriever.get_relevant_documents(
    "What was mentioned in the 4th document about  Python")
print(docs)