import os
from fastmcp import FastMCP
import chromadb
from fastembed import TextEmbedding
from chromadb import Documents, EmbeddingFunction, Embeddings

class FastEmbedFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [embedding.tolist() for embedding in self.model.embed(input)]

mcp = FastMCP("FastSemanticNotes")
client = chromadb.PersistentClient(path="./fast_notes_db")

fast_ef = FastEmbedFunction()
collection = client.get_or_create_collection(
    name="private_knowledge", 
    embedding_function=fast_ef
)

@mcp.tool()
async def save_note(content: str) -> str:
    """Saves a note but blocks it if a similar one exists."""
    
    results = collection.query(
        query_texts=[content],
        n_results=1,
        include=['distances', 'documents']
    )

    if not results['documents'][0]:
        collection.add(
            documents=[content],
            ids=[str(hash(content))]
        )
        return "Note saved successfully using local FastEmbed."
    
    if results['distances'] and results['distances'][0][0] < 0.35:
        existing = results['documents'][0][0]
        return (
            f"SEMANTIC DUPLICATE DETECTED\n"
            f"Existing Note: \"{existing}\"\n"
            "This covers the same topic. Should I merge or ignore?"
        )

    note_id = str(hash(content))
    collection.add(
        documents=[content],
        ids=[note_id]
    )
    return "Note saved successfully using local FastEmbed."

@mcp.tool()
async def search_notes(query: str) -> str:
    """Finds notes based on meaning, not just exact words."""
    
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    if not results['documents'][0]:
        return "No relevant notes found in your history."

    formatted_notes = "\n".join([f"• {doc}" for doc in results['documents'][0]])
    return f"Relevant Context Found:\n{formatted_notes}"

@mcp.tool()
async def delete_note(note_id: str) -> str:
    """Deletes a note by its ID."""
    collection.delete(ids=[note_id])
    return "Note deleted successfully."

@mcp.tool()
async def list_notes() -> str:
    """Lists all notes in the collection."""
    notes = collection.get()
    formatted_notes = "\n".join([f"• {doc}" for doc in notes['documents']])
    return f"Notes in your history:\n{formatted_notes}"

if __name__ == "__main__":
    mcp.run()