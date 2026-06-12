import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from pinecone import Pinecone
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


#Initialize Embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small",)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
#Initialize vector store
vectorstore = PineconeVectorStore(
    index=pc.Index(os.environ["INDEX_NAME"]), embedding=embeddings
)

#Initialize Chat Model
llm = init_chat_model("gpt-4o", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    #Serialize document for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source',' Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model=llm, tools=[retrieve_context], system_prompt=system_prompt)

    #Build messages list 
    messages = [{"role": "user", "content":query}]

    #Invoke the agent
    response = agent.invoke({"messages": messages})

    #Extract the answer from the last AI message
    answer = response["messages"][-1].content

    #Extract context documents from ToolMessage artifacts
    content_docs = []
    for message in response["messages"]:
        #Check if this is a ToolMEssage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            #The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                content_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": content_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    print("tt")