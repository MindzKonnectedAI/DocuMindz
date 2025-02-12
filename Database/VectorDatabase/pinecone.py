from PDFParser.PyMuPDF4LLM.parse_pdf import parse_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text splitter ( TextSplitter to split Markdown )
from langchain_experimental.text_splitter import SemanticChunker
from Models.embedding import embeddings
import streamlit as st
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
import os
from pinecone import Pinecone, ServerlessSpec 
import time

# Text and Image summary creator functions
from ResponseGenerator.summaries import create_text_summaries,create_image_summaries

# Doc Store
from DocStore.docstore import get_mongo_docstore

from PDFParser.PyMuPDF.extract_pdf_images import convert_image_array_to_documents

from langchain_pinecone import PineconeVectorStore  # Langchain's Pinecone library

from Utils.session_states import initialize_session_states

from dotenv import load_dotenv  # .env file loading

load_dotenv(override=True)

# Initialize session states
initialize_session_states()

# Pinecone setup (for vector storage)
api_key_pinecone = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key_pinecone)

def list_existing_indexes():
    indexes = pc.list_indexes()
    return indexes

def wait_on_namespace(index_name: str, namespace: str):
    """
    Waits until the specified namespace is available in the Pinecone index.
    """
    ready = False
    while not ready:
        try:
            index = pc.Index(index_name)
            index_stats = index.describe_index_stats()
                
            # Check if namespace exists in the index stats
            if namespace in index_stats.get("namespaces", {}):
                return True
        except pc.core.client.exceptions.NotFoundException:
            # If namespace isn't found, keep waiting
            pass
        time.sleep(5)

def wait_on_index(index: str):
    """
    Takes the name of the index to wait for and blocks until it's available and ready.
    """
    ready = False
    while not ready:
        try:
            desc = pc.describe_index(index)
            if desc[7]['ready']:
                return True
        except pc.core.client.exceptions.NotFoundException:
            # NotFoundException means the index is created yet.
            pass
        time.sleep(5)

def getVectorStore():
    if st.session_state.namespace and st.session_state.namespace!="Default":
        vectorstore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings,namespace=st.session_state.namespace)
        return vectorstore
    else:
        vectorstore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
        return vectorstore


# Create vector database for multiple files
def create_vector_database(user_folder, file_paths,selected_files):
    """
    Creates a vector database using document loaders and embeddings for multiple files.

    This function loads PDF documents,
    splits the loaded documents into chunks, transforms them into embeddings using OpenAIEmbeddings,
    and finally persists the embeddings into a Pinecone vector database.
    """
    try:
        print("Inside create_vector_database function")
            
        for file_path, file_name in zip(file_paths, selected_files):
            
            if file_name.endswith(".pdf"):
                print(f"Processing PDF file_type: {file_name}")
                
                parsed_data, unique_xref_array = parse_pdf(file_path, user_folder)
                # print ('parsed_Data', parsed_data)
                # print ('unique_xref_Array', unique_xref_array)
                docs = []

                if(st.session_state.chunking_strategy=="Recursive"):
                    ## Recursive Chunking 
                    recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
                    chunked_texts = recursive_text_splitter.split_text(parsed_data)
                    
                elif(st.session_state.chunking_strategy=="Semantic"):
                    ## Semantic Chunking
                    semantic_text_splitter = SemanticChunker(embeddings=embeddings,breakpoint_threshold_amount=85)
                    chunked_texts = semantic_text_splitter.split_text(parsed_data)

                # Convert chunks to LangChain Document objects
                # docs = [Document(page_content=text, metadata={"source": file_name}) for text in chunked_texts]
                docs = [Document(page_content=text) for text in chunked_texts]

                text_summaries= create_text_summaries(docs) #-> return text summaries 
                print ('length of text summary', len(text_summaries)) 
                # print ('this is text summary',text_summaries) 

                image_summaries= create_image_summaries(unique_xref_array) #-> return image summaries
                print ('length of image summary', len(image_summaries))
                # print ('this is image summary', image_summaries)
            
            elif file_name.endswith(".md"):
                # New logic for Markdown files
                print(f"Processing Markdown file: {file_name}")
                
                with open(file_path,"r",encoding="utf-8") as md_file:
                    md_content =md_file.read()
                    print("type is ",type(md_content))
                    print(f"Mardown content;\n{md_content}")
                    
                if(st.session_state.chunking_strategy=="Recursive"):
                    ## Recursive Chunking 
                    recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
                    chunked_texts = recursive_text_splitter.split_text(md_content)
                    
                elif(st.session_state.chunking_strategy=="Semantic"):
                    ## Semantic Chunking
                    semantic_text_splitter = SemanticChunker(embeddings=embeddings,breakpoint_threshold_amount=85)
                    chunked_texts = semantic_text_splitter.split_text(md_content)
                
                docs = [Document(page_content=text) for text in chunked_texts]
                
                print("Rucursive splitter ",docs)
                
                
                # docs = [Document(page_content=md_content)]
                text_summaries = create_text_summaries(docs)
                
                print('Length of text summary:', len(text_summaries))

                # No image summaries for Markdown files
                image_summaries=[]
               
            else : 
                print(f"Unsupported file type: {file_name}")
                continue  
                # Pinecone setup (for vector storage)
            vectorstore = getVectorStore()
                
                # The storage layer for the parent documents
            id_key = "doc_id"
                    
                # The retriever (empty to start)
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=get_mongo_docstore(st.session_state.index_name),
                id_key="doc_id",
                )
            if text_summaries:
                    # Add texts
                doc_ids = [str(uuid.uuid4()) for _ in docs]
                summary_texts = [
                     Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
                ]
                retriever.vectorstore.add_documents(summary_texts)
                retriever.docstore.mset(list(zip(doc_ids, docs)))

            if image_summaries:  
                    final_array = convert_image_array_to_documents(unique_xref_array)
                    # Add image summaries
                    img_ids = [str(uuid.uuid4()) for _ in final_array]
                    summary_img = [
                        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
                    ]
                    retriever.vectorstore.add_documents(summary_img)
                    retriever.docstore.mset(list(zip(img_ids, final_array)))  
                        
            print(file_name+" upserted to Pinecone successfully")
            return
    except Exception as e:
            print(f"Error details: {str(e)}")
            st.error(f"An error occurred while creating the vector database: {e}")

    
def process_selected_files(save_folder, email,selected_files):
    try:
        file_paths = []
        for file in selected_files:
            file_path = os.path.join(save_folder, file)
            file_paths.append(file_path)
            
        # Check if the index exists
        existing_indexes = list_existing_indexes()

        if not any(index.name == st.session_state.index_name for index in existing_indexes):
            print("Creating new index")
            # Create a new index if it doesn't already exist
            pc.create_index(
                name=st.session_state.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait_on_index(st.session_state.index_name)
        else:
            print(f"Index already exists: {st.session_state.index_name}")

        # Create user-specific directory in data/
        user_folder = os.path.join("data", email,st.session_state.namespace)
        os.makedirs(user_folder, exist_ok=True)
            
        # Create the vector database for multiple files
        create_vector_database(user_folder, file_paths, selected_files)
            
        # Save the names of the files that were converted
        selected_file_folder = os.path.join("selected", email,st.session_state.namespace)
        os.makedirs(selected_file_folder, exist_ok=True)
        text_file_path = os.path.join(selected_file_folder, "selected.txt")
            
        with open(text_file_path, "w") as f:
            for file_name in selected_files:
                f.write(file_name + "\n")
            
        print("Successfully processed all file")
        return True
    except  Exception as e:
        print(f"Error in process_selected_files: {str(e)}")
        st.error(f"An error occurred while processing files: {str(e)}")
        return False
