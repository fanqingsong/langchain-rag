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
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

def documentLoader():
    """Load all documents from the pdfs directory"""
    return PyPDFDirectoryLoader(DOCUMENTS_PATH).load()

def textSplitter(documents: Iterable[Document]):
    """Split the text of each document into characters"""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    ).split_documents(documents)

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
    embedder = ollamaEmbedder()

    # Init and load the Chroma vector store
    db = initChromaDB(embedder)
    updateChromaDB(db, chunks)

    # Usage example
    model = ollamaLLM()
    userInput = ""

    print("You can make questions about the document you introduced.")
    print("If you want to use more documents, restart the app after.")
    print("Exit by typing 'exit'.")
    while userInput != "exit":
        userInput = input("> ")
        context, sources = searchDBForQueryContext(userInput, db)
        prompt = formatPrompt(context, userInput)
        response = model.invoke(prompt)
        print(response)
        print("Sources:" + str(sources))