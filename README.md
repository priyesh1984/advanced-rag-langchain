# LangChain Demo

A collection of LangChain.js experiments exploring RAG (Retrieval-Augmented Generation), AI agents, and LangGraph workflows using Ollama (local LLMs) and Pinecone (vector database).

## Projects

### 1. Basic LLM Chain (`index.js`)

Generates a structured 2-day travel itinerary for a city using a local Ollama model and LangChain's prompt + parser pipeline.

### 2. RAG from File + Pinecone (`rag-ollama-pinecone.js`)

Extends RAG to work with file content (e.g., a PDF/CV):

- Reads a file and splits it into chunks using `RecursiveCharacterTextSplitter`
- Upserts chunks in batches to Pinecone
- Queries Pinecone with a user question and streams the final answer via Ollama

### 3. CV Chat Server (`server.js`)

An Express.js server with a `/api/chat` endpoint that:

- Accepts user questions about a CV stored in Pinecone
- Retrieves the top 6 relevant chunks, sorted by their original document order
- Streams the AI response back to the frontend using Server-Sent Events (SSE)
- Serves a static frontend from the `public/` directory

## Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434` with `llama3` pulled
- A [Pinecone](https://www.pinecone.io/) account with an index configured

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Create a `.env` file in the project root:

   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   INDEX_NAME=your_index_name
   INDEX_HOST=your_index_host_url
   ```

3. Pull required Ollama models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

## Running

```bash
# Basic Ollama chain
node index.js

# RAG from file
node rag-ollama-pinecone.js

# CV Chat server (port 3001)
node server.js

```

## Tech Stack

| Tool                                                     | Purpose                                        |
| -------------------------------------------------------- | ---------------------------------------------- |
| [LangChain.js](https://js.langchain.com/)                | Prompt chains, output parsers, text splitters  |
| [LangGraph](https://langchain-ai.github.io/langgraphjs/) | Stateful multi-step agent workflows            |
| [Ollama](https://ollama.com/)                            | Local LLM inference (llama3, nomic-embed-text) |
| [Pinecone](https://www.pinecone.io/)                     | Vector database for semantic search            |
| [Express.js](https://expressjs.com/)                     | HTTP server for chat API endpoints             |
