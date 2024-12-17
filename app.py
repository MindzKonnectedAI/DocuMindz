import os
from typing import Literal
from typing import Annotated, Sequence, TypedDict
import shutil
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from llm import llmmodel
from pydantic import BaseModel,Field
# from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv  # .env file loading
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Environment variables setup
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Agentic Rag"
os.environ["USER_AGENT"] = "MyAgent/1.0"


# Function to initialize retriever
def initialize_retriever(upload_folder="uploaded_pdfs", persist_dir="./db"):
    # Delete the old database directory if it exists
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # Remove the directory and its contents
        print(f"Old database at '{persist_dir}' has been deleted.")
    

    if not os.path.exists(upload_folder) or not os.listdir(upload_folder):
        print(f"No files found in '{upload_folder}'. Skipping retriever initialization.")
        return None  # Return None if no files are found
    
    # Initialize the document list
    docs = []
    uploaded_pdf_files = os.listdir(upload_folder)
    
    for pdf_file in uploaded_pdf_files:
        file_path = os.path.join(upload_folder, pdf_file)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Create a new Chroma vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(api_key=openai_api_key),
        persist_directory=persist_dir,
    )
    
    # Persist the new database
    vectorstore.persist()
    print(f"New database created and persisted at '{persist_dir}'.")

    # Load the retriever from the newly created database
    retriever = Chroma(
        persist_directory=persist_dir,
        collection_name="rag-chroma",
        embedding_function=OpenAIEmbeddings(api_key=openai_api_key)
    ).as_retriever()

    return retriever


graph = None

# Use your own Pinecone DB instance
def ProcessDoucment():
    global graph
    retriever = initialize_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_document",
        description="Search and return relevant information",
    )

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        model = llmmodel
        llm_with_tool = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template=(
                """You are a grader assessing relevance of a retrieved document to a user question.\n\n"
                "Here is the retrieved document:\n\n{context}\n\n"
                "Here is the user question: {question}\n"
                "Grade as 'yes' or 'no' based on relevance."""
            ),
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        return "generate" if score == "yes" else "rewrite"


    # Define agent
    def agent(state):
        messages = state["messages"]
        model = llmmodel.bind_tools([retriever_tool])
        
        try:
            response = model.invoke(messages)
        except Exception as e:
            print("Error invoking retriever tool:", str(e))
            response = {"text": "No relevant information available in the database."}

        return {"messages": [response]}


    # Rewrite the query
    def rewrite(state):
        question = state["messages"][0].content

        msg = [
            HumanMessage(
                content=(
                    """
                    Look at the input and try to reason about the underlying semantic intent.\n
                    Here is the initial question:\n-------\n{question}\n-------\n
                    Formulate an improved question.
                    """.format(question=question)
                )
            )
        ]
        model = llmmodel
        response = model.invoke(msg)
        return {"messages": [response]}


    # Generate response
    def generate(state):
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = hub.pull("rlm/rag-prompt")
        llm = llmmodel

        rag_chain = prompt | llm | StrOutputParser()

        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]
    # Define workflow graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile graph
    graph = workflow.compile()
    



# Main function to generate a response
def response_generator(prompt):
    global graph
    inputs = {"messages": [("user", prompt)]}
    response = graph.invoke(inputs)
    return response["messages"][-1].content
