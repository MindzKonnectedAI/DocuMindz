from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from cohere.client import Client as CohereClient
from langchain_cohere import CohereRerank  # CohereRerank for reranking
from langchain_pinecone import PineconeVectorStore  # Langchain's Pinecone library
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from typing import Literal
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI # OpenAI's Chat Model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Cohere setup (for reranking)
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = CohereClient(api_key=cohere_api_key)
# OpenAI Embeddings setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# OpenAI setup
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=True,
)

# def reRanker():
#     compressor = CohereRerank(model="rerank-english-v3.0",client=cohere_client)
#     vectorStore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
#     compression_retriever = ContextualCompressionRetriever(
#         base_compressor=compressor,
#         base_retriever=vectorStore.as_retriever(
#             search_kwargs={"k": 5},
#         ),
#     )
#     return compression_retriever

# Main function to generate a response
def generate_response(prompt,index_name,chat_history):
    vectorStore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_document",
        description="Search and return relevant information",
    )

    def grade_documents(state) -> Literal["generate", "rewrite"]:
        # print("inside grade_documents")
        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        model = llm
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
        # print("type of docs :",type(docs))
        # print("documents for grading :",docs)
        scored_result = chain.invoke({"question": question, "context": docs})
        # print("scored_result :",scored_result)
        score = scored_result["binary_score"]
        print("binary_score :",score)

        return "generate" if score == "yes" else "rewrite"


    # Define agent
    def agent(state):
        # print("inside agent")
        messages = chat_history + state["messages"]
        # print("messages of agent :",messages)
        model = llm.bind_tools([retriever_tool])
            
        try:
            response = model.invoke(messages)
            # print("agent's response :",response)
        except Exception as e:
            print("Error invoking retriever tool:", str(e))
            response = {"text": "No relevant information available in the database."}

        return {"messages": [response]}

    # Rewrite the query
    def rewrite(state):
        # print("inside rewrite")
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
        model = llm
        response = model.invoke(msg)
        # print("response of rewrite :",response)
        return {"messages": [response]}

    def generate(state):
        # print("inside generate")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        # print("type of docs :",type(docs))
        # print("docs that generate is going to use :",docs)
        # Define the custom prompt directly
        prompt_text = """
        You are a specialized document analysis assistant designed to provide precise, context-rich answers by synthesizing information from tables, 
        structured text, and visual elements within provided PDF documents. You also possess advanced mathematical reasoning and calculation capabilities.

        When responding to questions, adhere to the following guidelines:

        1. Table Data Analysis
        - Extract and format numerical values and relationships into clear, tabular layouts
        - Always:
            * Specify table titles and numbers for source identification
            * Include footnotes or special notations to preserve context
        - For tables spread across multiple pages:
            * Clearly indicate when a table spans multiple pages
            * Consolidate the data into a single cohesive format, ensuring no information is missed
            * Reference the page range where the table appears for clarity
        - For mathematical operations involving table data:
            * Present the relevant table data
            * Show each calculation step clearly with labeled subtotals and intermediate results
            * Validate results by cross-referencing multiple tables if applicable

        2. Mathematical Reasoning and Calculations
        Steps to Perform Calculations:
        - Clearly define the mathematical problem or objective
        - List all relevant data with precise source references (e.g., page numbers, table numbers)
        - Show every calculation step with detailed explanations using proper notation and units
        - Validate consistency of units and formats before proceeding
        - Verify results through cross-checking or secondary calculations
        - Present the final answer with appropriate context

        For calculations across multiple tables:
        - Organize all relevant data in a structured format
        - Show relationships between different data sources
        - Clearly explain assumptions and data transformations

        3. Visual Content Interpretation
        Analyze charts and graphs:
        - Describe data values, trends, and patterns, referencing axes, legends, and scales
        - Extract numerical data as needed for calculations
        - Summarize findings by connecting visual data with related text or tables

        4. Textual Information
        - Reference sections and page numbers when quoting or summarizing text
        - Retain original formatting (e.g., bullet points, numbered lists, paragraphs)
        - Capture hierarchical details, including footnotes and cross-references
        - Extract numerical information for calculations when relevant

        Response Format Guidelines:
        - Source Identification: Start by identifying data sources (e.g., table, text, visual)
        - Tables: Present data in a clean table format for readability
        - Calculations:
            * Use markdown code blocks for showing calculation steps
            * Clearly format equations with intermediate results and units
        - Text: Preserve PDF-style formatting (e.g., bullets, lists)
        - Visuals: Summarize data with references to legends, axes, and scales
        - Locations: Cite exact locations (page numbers, section titles) for all referenced information
        - Cross-Referencing: Connect related document elements for a cohesive response
        - Data Integrity: Maintain the original precision, units, and context of all data

        Mathematical Operations Format:
        Step 1: Define the objective
        Step 2: List source data with references
        Step 3: Show the calculation setup
        Step 4: Perform step-by-step operations
        Step 5: Verify results
        Step 6: Present the final result with context

        Error Handling:
        If the required information is not found in the documents, respond with:
        "I cannot locate specific information about this in the provided PDF documents. Please verify if this information is included or consider rephrasing your question."

        For tables spanning multiple pages, provide a consolidated analysis of the data across those pages, ensuring completeness and accuracy.

        You may respond to basic greetings, but for all other queries, strictly adhere to the provided document content.

        Context: {context}
        Question: {question}
        Answer:
        """

        # Replace the text in the prompt object
        prompt = hub.pull("rlm/rag-prompt")
        prompt.messages[0].prompt.template = prompt_text

        # print("prompt logged here:", prompt)

        # Invoke the chain
        rag_chain = prompt | llm | StrOutputParser()
        response = rag_chain.invoke({"context": docs, "question": question})
        # print("generate function's response :",response)
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


    inputs = {"messages": [("user", prompt)]}
    response = graph.invoke(inputs)
    return response["messages"][-1].content
