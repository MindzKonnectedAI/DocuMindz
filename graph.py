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
def generate_response(prompt,index_name):
    vectorStore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve_document",
        description="Search and return relevant information",
    )

    def grade_documents(state) -> Literal["generate", "rewrite"]:
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

        scored_result = chain.invoke({"question": question, "context": docs})
        print("scored_result :",scored_result)
        # score = scored_result.binary_score
        score = scored_result["binary_score"]
        print("binary_score :",score)

        return "generate" if score == "yes" else "rewrite"


    # Define agent
    def agent(state):
        messages = state["messages"]
        model = llm.bind_tools([retriever_tool])
            
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
        model = llm
        response = model.invoke(msg)
        return {"messages": [response]}


    # Generate response
    def generate(state):
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = hub.pull("rlm/rag-prompt")

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


    inputs = {"messages": [("user", prompt)]}
    response = graph.invoke(inputs)
    return response["messages"][-1].content
