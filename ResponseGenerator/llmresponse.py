from pydantic.v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from Models.chat import llm
import streamlit as st
from Models.rerank import cohere_reranker
from Database.VectorDatabase.pinecone import getVectorStore
from DocStore.docstore import get_mongo_docstore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import re
from operator import itemgetter
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from Utils.session_states import initialize_session_states
from Database.DocumentDatabase.dbscript import get_current_session_history
from Cache.llm_cache import get_mongo_cache
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from typing import Any, Callable, Dict, Optional, Union
from typing import Any, Dict, Optional, Sequence, Tuple
from langchain_core.outputs import Generation

# Initialize session states
initialize_session_states()

def parse_docs(docs,need_image= False):
    """
    Split base64-encoded images and texts.
        
    Args:
        docs (list): A list of objects with the `page_content` attribute.
            
    Returns:
        dict: A dictionary with two keys: 
            "images" containing base64-encoded strings,
            "texts" containing textual content.
    """
    print(f"Received {len(docs)} documents for parsing.")

    base64_pattern = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')
    b64_images = []
    text_contents = []
        
    for doc in docs:
        if not hasattr(doc, 'page_content'):
            print(f"Skipping document without 'page_content': {doc}")
            continue
            
        content = doc.page_content.strip()
            
        # Check if the content looks like base64
        if base64_pattern.fullmatch(content):
            try:
                base64.b64decode(content, validate=True)
                b64_images.append(content)  # Valid base64 image
                continue
            except Exception as e:
                print(f"Base64 decoding failed for content: {content[:30]}... Error: {e}")
            
        # If not base64 or decoding fails, treat it as text
        text_contents.append(content)
        
    print(f"Parsed {len(b64_images)} images and {len(text_contents)} texts.")
    if need_image:
        return {"images": b64_images, "texts": text_contents}
    else:
        return {"images": [], "texts": text_contents}

def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]


    # context_text = ""
    # if len(docs_by_type["texts"]) > 0:
    #     for text_element in docs_by_type["texts"]:
    #         context_text += text_element

    # # construct prompt with context (including images)
    # prompt_template = f"""
    #     You are a specialized document analysis assistant designed to provide precise, context-rich answers by synthesizing information from tables, 
    #     structured text, and visual elements within provided PDF documents. You also possess advanced mathematical reasoning and calculation capabilities.
    #     If the required information is not found in the documents, respond with:
    #     "I cannot locate specific information about this in the provided PDF documents. Please verify if this information is included or consider rephrasing your question."
    #     You may respond to basic greetings, but for all other queries, strictly adhere to the provided document content.
    #     \n\n
    #     **Context** : {context_text}
    # """

    # Safely join text without using f-strings
    context_text = "".join(docs_by_type["texts"]) if docs_by_type["texts"] else ""
    
    # Ensure special characters like `{}`, `[]`, `<`, `&` do not break the prompt
    context_text = context_text.replace("{", "{{").replace("}", "}}")

    # construct prompt with context (including images)
    # prompt_template = """ 
    #     You are a specialized document analysis assistant designed to provide precise, context-rich answers by synthesizing information from tables, 
    #     structured text, and visual elements within provided PDF documents. You also possess advanced mathematical reasoning and calculation capabilities.
    #     If the required information is not found in the documents, respond with:
        
    #     "I cannot locate specific information about this in the provided PDF documents. Please verify if this information is included or consider rephrasing your question."
        
    #     You may respond to basic greetings, but for all other queries, strictly adhere to the provided document content.

    #     **Context** : {context}
    # """.format(context=context_text)
    prompt_template = """  
            You are a highly specialized document analysis assistant with expertise in extracting and synthesizing information from structured text, tables, and visual elements within provided PDF documents. Additionally, you possess advanced mathematical reasoning and calculation capabilities.  

            **Guidelines for Responses:**  
            - Your answers should be precise, context-rich, and strictly based on the provided document content.  
            - If the required information is not found in the documents, respond with:  

            *"I cannot locate specific information about this in the provided PDF documents. Please verify if this information is included or consider rephrasing your question."*  

            - You may respond to basic greetings, but for all other queries, adhere strictly to the document content.  

            **Context:**  
            {context}  
            """.format(context=context_text)

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            ("system",prompt_template),
            MessagesPlaceholder("chat_history"),
            ("human",user_question),
        ]
    )

def display_base64_image_in_streamlit(base64_code):
    """
    Display a base64-encoded image in a Streamlit app.

    Parameters:
    - base64_code (str): Base64-encoded string of the image
    """
    try:
        # Decode the base64 string to binary
        image_data = base64.b64decode(base64_code)
        # Convert binary to a format Streamlit can display
        st.image(image_data, caption="Referenced Image")
    except Exception as e:
        st.error(f"Error displaying the image: {e}")

def generations_to_string(generations: Optional[Sequence[Generation]]) -> str:
    if generations is None:
        return ""  # Return an empty string if no result found
    
    return " ".join(gen.text for gen in generations)

def generate_response(prompt: str,llm_string: str) :
    try:
        mongo_cache = get_mongo_cache()
        lookupResponse = mongo_cache.lookup(prompt,llm_string)
        if lookupResponse:
            chat_history = get_current_session_history()
            chat_history.add_user_message(prompt)
            ai_message = generations_to_string(lookupResponse)
            chat_history.add_ai_message(ai_message)
            return ai_message
        else:
            class ImageRequirementResponse(BaseModel):
                Need_image: bool = Field(description="Classification of the query as 'text', 'image', or 'both'")

            parser = JsonOutputParser(pydantic_object=ImageRequirementResponse)

            def classify_query_needs_image(prompt: str) -> str:
                """Classifies whether the query requires an image or not."""
                classifier_prompt = PromptTemplate(
                    # template="""
                    # You are an AI classifier. Your task is to determine if the given query requires an image in the response. 
                    # An image is needed if the query mentions or implies visual elements such as diagrams, pictures, logos, maps, 
                    # or asks about how something looks, appears, or is represented visually.

                    # {format_instructions}

                    # Query: "{prompt}"
                    # """,
                    template="""
                            You are an AI classifier. Your task is to determine whether the given query requires:
                            - Only text (if the query asks for explanations, descriptions, or non-visual answers).
                            - Only an image (if the query explicitly asks for a picture, diagram, graph, or any other visual representation).
                            - Both text and image (if the query requires a combination of explanation and visual representation).

                            Return one of the following: "text", "image", or "both".
                            {format_instructions}

                            Query: "{prompt}"
                            Classification:
                            """,
                    input_variables=["prompt"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                chain = classifier_prompt | llm | parser
                result = chain.invoke({"prompt": prompt})
                print("result :",result)
                return result["Need_image"]

            need_image = classify_query_needs_image(prompt)     

            contextualize_q_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

            # Reranker 
            def reRanker():
                vectorStore = getVectorStore()
                    
                retriever = MultiVectorRetriever(
                    vectorstore=vectorStore,
                    docstore=get_mongo_docstore(st.session_state.index_name),
                    id_key="doc_id",
                )

                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=cohere_reranker,
                    base_retriever=retriever,
                )

                return compression_retriever

            compression_retriever = reRanker()

            history_aware_retriever = create_history_aware_retriever(
                llm, compression_retriever, contextualize_q_prompt
            )

            chain_with_sources = {
                "context": history_aware_retriever | RunnableLambda(lambda docs: parse_docs(docs, need_image=need_image)), # {"images": b64_images, "texts": text_contents}
                "question": itemgetter("input"),
                "chat_history": itemgetter("chat_history"), 
            } | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(build_prompt)
                    | llm
                    | StrOutputParser()
                )
            )
            MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")
            
            def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
                print("inside session history ",session_id)
                return MongoDBChatMessageHistory(
                    MONGO_DB_CONN_STR , session_id, database_name="new", collection_name="history"
                )
            
            with_message_history = RunnableWithMessageHistory(
                chain_with_sources, 
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="response",
            )
            dossier_session_id=st.session_state.namespace+"_"+st.session_state.session_id
            print("dossier_session_id :",dossier_session_id)

            answer = with_message_history.invoke({"input":prompt},{"configurable": {"session_id":dossier_session_id }},)
            
            for image in answer['context']['images']:
                display_base64_image_in_streamlit(image)
            return answer["response"]
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")



    # # generate response 
    # def generate_response(prompt: str) :
    #     try:
    #         contextualize_q_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    #         # Reranker 
    #         def reRanker():
    #             compressor = CohereRerank(model="rerank-english-v3.0",client=cohere_client)
    #             vectorStore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
                
    #             id_key = "doc_id"
    #             docstore = MongoDBStore(MONGO_DB_CONN_STR, db_name="new",collection_name=st.session_state.index_name)
                
    #             retriever = MultiVectorRetriever(
    #                 vectorstore=vectorStore,
    #                 docstore=docstore,
    #                 id_key=id_key,
    #             )

    #             compression_retriever = ContextualCompressionRetriever(
    #                 base_compressor=compressor,
    #                 base_retriever=retriever,
    #             )

    #             return compression_retriever

    #         compression_retriever = reRanker()

    #         history_aware_retriever = create_history_aware_retriever(
    #                 llm, compression_retriever, contextualize_q_prompt
    #         )

            # system_prompt = """You are a specialized document analysis assistant designed to provide precise, context-rich answers by synthesizing information from tables, 
            # structured text, and visual elements within provided PDF documents. You also possess advanced mathematical reasoning and calculation capabilities.

            # When responding to questions, adhere to the following guidelines:

            # 1. Table Data Analysis
            # - Extract and format numerical values and relationships into clear, tabular layouts
            # - Always:
            #     * Specify table titles and numbers for source identification
            #     * Include footnotes or special notations to preserve context
            # - For tables spread across multiple pages:
            #     * Clearly indicate when a table spans multiple pages
            #     * Consolidate the data into a single cohesive format, ensuring no information is missed
            #     * Reference the page range where the table appears for clarity
            # - For mathematical operations involving table data:
            #     * Present the relevant table data
            #     * Show each calculation step clearly with labeled subtotals and intermediate results
            #     * Validate results by cross-referencing multiple tables if applicable

            # 2. Mathematical Reasoning and Calculations
            # Steps to Perform Calculations:
            # - Clearly define the mathematical problem or objective
            # - List all relevant data with precise source references (e.g., page numbers, table numbers)
            # - Show every calculation step with detailed explanations using proper notation and units
            # - Validate consistency of units and formats before proceeding
            # - Verify results through cross-checking or secondary calculations
            # - Present the final answer with appropriate context

            # For calculations across multiple tables:
            # - Organize all relevant data in a structured format
            # - Show relationships between different data sources
            # - Clearly explain assumptions and data transformations

            # 3. Visual Content Interpretation
            # Analyze charts and graphs:
            # - Describe data values, trends, and patterns, referencing axes, legends, and scales
            # - Extract numerical data as needed for calculations
            # - Summarize findings by connecting visual data with related text or tables

            # 4. Textual Information
            # - Reference sections and page numbers when quoting or summarizing text
            # - Retain original formatting (e.g., bullet points, numbered lists, paragraphs)
            # - Capture hierarchical details, including footnotes and cross-references
            # - Extract numerical information for calculations when relevant

            # Response Format Guidelines:
            # - Source Identification: Start by identifying data sources (e.g., table, text, visual)
            # - Tables: Present data in a clean table format for readability
            # - Calculations:
            #     * Use markdown code blocks for showing calculation steps
            #     * Clearly format equations with intermediate results and units
            # - Text: Preserve PDF-style formatting (e.g., bullets, lists)
            # - Visuals: Summarize data with references to legends, axes, and scales
            # - Locations: Cite exact locations (page numbers, section titles) for all referenced information
            # - Cross-Referencing: Connect related document elements for a cohesive response
            # - Data Integrity: Maintain the original precision, units, and context of all data

            # Mathematical Operations Format:
            # Step 1: Define the objective
            # Step 2: List source data with references
            # Step 3: Show the calculation setup
            # Step 4: Perform step-by-step operations
            # Step 5: Verify results
            # Step 6: Present the final result with context

            # Error Handling:
            # If the required information is not found in the documents, respond with:
            # "I cannot locate specific information about this in the provided PDF documents. Please verify if this information is included or consider rephrasing your question."

            # For tables spanning multiple pages, provide a consolidated analysis of the data across those pages, ensuring completeness and accuracy.

            # You may respond to basic greetings, but for all other queries, strictly adhere to the provided document content.

            # {context}"""


    #         chatPrompt = ChatPromptTemplate.from_messages(
    #                 [
    #                     ("system", system_prompt),
    #                     MessagesPlaceholder("chat_history"),
    #                     ("human", "{input}"),
    #                 ]
    #         )
                            
    #         question_answer_chain = create_stuff_documents_chain(llm, chatPrompt)

    #         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    #         conversational_rag_chain = RunnableWithMessageHistory(
    #                 rag_chain,
    #                 get_session_history,
    #                 input_messages_key="input",
    #                 output_messages_key="answer",
    #                 history_messages_key="chat_history",
    #         )
    #         for chunk in conversational_rag_chain.stream(input={"input": prompt},config={'configurable': {'session_id': st.session_state.session_id}}):
    #             answer_chunk = chunk.get("answer")
    #             if answer_chunk:
    #                 yield answer_chunk
    #     except Exception as e:
    #         st.error(f"An error occurred while generating the response: {e}")

