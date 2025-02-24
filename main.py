# Variables
DOCUMENTS_PATH = "./pdfs"
CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """
ANSWER THE QUESTION BASED ONLY ON THE FOLLOWING CONTEXT:
{context}

---
ANSWER THE FOLLOWING QUESTION BASED ONLY ON THE CONTEXT ABOVE: {question}
"""

# Type imports
from typing import Iterable
from chromadb import Embeddings

# Imports
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

import requests



def documentLoader():
    """Load all documents from the pdfs directory"""
    return PyPDFDirectoryLoader(DOCUMENTS_PATH).load()

def textSplitter(documents: Iterable[Document]):
    """Split the text of each document into characters"""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    ).split_documents(documents)

from typing import List

from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    """ParrotLink embedding model integration.

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of ParrotLink model to use.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_parrot_link import SiliconFlowEmbeddings

            embed = SiliconFlowEmbeddings(
                model="...",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            embed.embed_query(input_text)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Embed multiple text:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            embed.embed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await embed.aembed_query(input_text)

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            # TODO: Example output.

    """

    def __init__(self, model: str):
        self.model = model

    def embed_one_sentence(self, text: str) -> List[float]:
        """Embed a single sentence."""
        url = "https://api.siliconflow.cn/v1/embeddings"

        payload = {
            "model": "BAAI/bge-large-zh-v1.5",
            "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!",
            "encoding_format": "float"
        }
        headers = {
            "Authorization": "Bearer sk-lnslzrriteyazhaiirvvuawzgtsfsflpdhuxphqkfzacppdz",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        print(response.text)

        json_response = response.json()

        return json_response['data'][0]['embedding']


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [self.embed_one_sentence(one_sentence) for one_sentence in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    # optional: add custom async implementations here
    # you can also delete these, and the base class will
    # use the default implementation, which calls the sync
    # version in an async executor:

    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Asynchronous Embed search docs."""
    #     ...

    # async def aembed_query(self, text: str) -> List[float]:
    #     """Asynchronous Embed query text."""
    #     ...


def siliconflowEmbedder():
    """Embed the characters of each document"""
    return SiliconFlowEmbeddings(model="")

def ollamaEmbedder():
    """Embed the characters of each document"""
    return OllamaEmbeddings(model="nomic-embed-text")

def ollamaLLM():
    """Create an Ollama language model object"""
    return Ollama(model="llama3.2")

def initChromaDB(embedder: Embeddings):
    """Create a Chroma vector store from the documents"""
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)

def chunkIdGen(chunks: list[Document]):
    """Create an id for a chunk based on document, page and position"""
    currSource = ""
    currPage = -1
    currPosition = -1
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        currPosition = currPosition + 1 if source == currSource and page == currPage else 0
        currSource = source
        currPage = page
        chunk.metadata["id"] = f"{source}:{page}:{currPosition}"

def updateChromaDB(db: Chroma, chunks: list[Document]):
    """Update a Chroma vector store with new documents"""
    dbItems = db.get(include=[])
    existingIds = set(dbItems["ids"])
    
    newChunks = [chunk for chunk in chunks if chunk.metadata.get("id") not in existingIds]
    chunkIdGen(newChunks)
    
    if len(newChunks) > 0:
        print(f"Adding {len(newChunks)} new chunks to the Chroma database")
        db.add_documents(newChunks, ids=[chunk.metadata.get("id") for chunk in newChunks])
        db.persist()

def searchDBForQueryContext(query: str, db: Chroma):
    """Query the Chroma vector store for contexts similar to the query"""
    results = db.similarity_search_with_score(query, k=5)
    return (
        "\n\n---\n\n".join([doc.page_content for doc, _score in results]),
        [doc.metadata.get("id", None) for doc, _score in results]
    )

def formatPrompt(context: str, question: str):
    """Format a prompt for the user"""
    return PROMPT_TEMPLATE.format(context=context, question=question)

if __name__ == "__main__":
    # Load the documents
    documents = documentLoader()

    # Split the text of each document into characters
    chunks = textSplitter(documents)

    # Load embedding function
    # embedder = ollamaEmbedder()
    embedder = siliconflowEmbedder()

    # Init and load the Chroma vector store
    db = initChromaDB(embedder)
    updateChromaDB(db, chunks)

    # Usage example
    # model = ollamaLLM()

    model = ChatOpenAI(
        openai_api_base="https://api.siliconflow.cn/v1", # 硅基流动的url
        openai_api_key="sk-lnslzrriteyazhaiirvvuawzgtsfsflpdhuxphqkfzacppdz",	# 自己的api-key
        model = "Qwen/Qwen2.5-7B-Instruct"  # 启用模型
    )
    userInput = ""

    print("You can make questions about the document you introduced.")
    print("If you want to use more documents, restart the app after.")
    print("Exit by typing 'exit'.")
    while True:
        userInput = input("> ")
        if userInput == "exit":
            break
        context, sources = searchDBForQueryContext(userInput, db)
        prompt = formatPrompt(context, userInput)
        response = model.invoke(prompt)
        print(response)
        print("Sources:" + str(sources))