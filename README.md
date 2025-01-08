# Langchain-RAG

Langchain-RAG is a Python project that demonstrates a Retrieval-Augmented Generation (RAG) system using Langchain, Chroma, and Ollama. This project loads PDF documents, splits them into chunks, embeds the chunks, stores them in a Chroma vector store, and allows querying the store to generate context-based answers using a language model.

## Requirements

- Python 3.13 or higher
- The dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create and activate a virtual environment (for MacOS):

    ```sh
    python -m venv env
    source env/bin/activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your PDF documents into a new ```./pdfs/``` directory.

2. Run the main script:

    ```sh
    python main.py
    ```

3. Follow the prompts to ask questions about the content of your PDF documents. You can make questions about the PDF documents you used earlier. Type `exit` to quit.

## Project Structure

- `main.py`: The main script that loads documents, splits text, embeds chunks, initializes the Chroma vector store, and handles user queries.
- `notebook.ipynb`: A Jupyter notebook with similar functionality for interactive use.
- `requirements.txt`: Lists the dependencies required for the project.
- `./pdfs/`: Directory to store PDF documents.

## Functions

### `documentLoader()`

Loads all documents from the `./pdfs/` directory using `PyPDFDirectoryLoader`.

### `textSplitter(documents: Iterable[Document])`

Splits the text of each document into chunks using `RecursiveCharacterTextSplitter`.

### `ollamaEmbedder()`

Creates an Ollama embeddings object for embedding document chunks.

### `ollamaLLM()`

Creates an Ollama language model object.

### `initChromaDB(embedder: Embeddings)`

Initializes a Chroma vector store with the given embedding function.

### `chunkIdGen(chunks: list[Document])`

Generates unique IDs for document chunks based on their source, page, and position.

### `updateChromaDB(db: Chroma, chunks: list[Document])`

Updates the Chroma vector store with new document chunks.

### `searchDBForQueryContext(query: str, db: Chroma)`

Queries the Chroma vector store for contexts similar to the given query.

### `formatPrompt(context: str, question: str)`

Formats a prompt for the language model using the given context and question.
