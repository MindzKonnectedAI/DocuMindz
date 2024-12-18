#  UI components
import streamlit as st # streamlit module ( for building UI )
import os # Operating System module
from dotenv import load_dotenv  # .env file loading

#  LLM and Chat Components
from langchain_openai import ChatOpenAI # OpenAI's Chat Model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# PDF Processing 
from langchain_community.document_loaders import UnstructuredMarkdownLoader # Markdown file loader ( because we're using LlamaParse )
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text splitter ( TextSplitter to split Markdown )

# Vectore Datebase & Embedding 
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec  # Pinecone as Vector DB (Pinecone's Python Library)
from langchain_pinecone import PineconeVectorStore  # Langchain's Pinecone library

# Reranking and Retrieval 
from langchain.retrievers import ContextualCompressionRetriever
from cohere.client import Client as CohereClient
from langchain_cohere import CohereRerank  # CohereRerank for reranking

# Chain Components
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

# Chat History Mangements
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Authentication & Database
import streamlit_authenticator_mongo as stauth
from dbscript import collection
from streamlit_authenticator_mongo.validator import Validator
from streamlit_authenticator_mongo.hasher import Hasher

# Langgraph Graph
from graph import generate_response

from langchain.globals import set_verbose
import nest_asyncio # not sure if this was needed in PDFRAG app
import yaml
from yaml.loader import SafeLoader
from datetime import datetime

import pymupdf4llm
from pathlib import Path

import json 
import time

def main():

    # Page Configuration
    st.set_page_config("DocuMindz",":bookmark_tabs:")

    # stauth package's validator
    validator = Validator()

    # config file of stauth package
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    # defaults
    load_dotenv(override=True)
    nest_asyncio.apply() # not sure if this was needed in PDFRAG app
    set_verbose(True) # removed the verbose warning by this 

    # clear console function
    def cls():
        os.system('cls' if os.name=='nt' else 'clear')

    # authenticator setup
    authenticator = stauth.Authenticate(
        collection,
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    def extract_images_from_pdf(file_path, user_folder):
        """ 
        Extract images from PDF with enhanced metadata using pymupdf4llm.
        Args:
        file_path (str): Path to the PDF file.
        user_folder (str): Path to the user's folder to save extracted images.
        
        Returns:
        list: List of dictionaries containing image details.
        """
        images = []
        
        # Create directory for extracted images
        images_dir = os.path.join(user_folder, 'extracted_images')
        os.makedirs(images_dir, exist_ok=True)

        try:
            # Convert PDF to Markdown while extracting images
            parsed_data = pymupdf4llm.to_markdown(
                    file_path,
                    write_images=True,
                    image_size_limit=0.001,  # Filter images based on size
                    margins=0,  # Remove margins
                    image_path=images_dir
                )

            # Process extracted images for metadata
            for img_file in Path(images_dir).glob("*.png"):
                img_filename = img_file.name
                img_path = str(img_file)
                file_size = os.path.getsize(img_path)
                
                # Collect image metadata
                images.append({
                    "filename": img_filename,
                    "path": img_path,
                    "file_size": file_size,
                    "extraction_timestamp": datetime.now().isoformat()
                })
                
            return images , parsed_data
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            
            return [], None
        
    # OpenAI setup
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        api_key=openai_api_key,
        temperature=0,
        model="gpt-4o-mini",
        streaming=True,
    )

    # Pinecone setup (for vector storage)
    api_key_pinecone = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key_pinecone)

    # Cohere setup (for reranking)
    cohere_api_key = os.getenv("COHERE_API_KEY")
    cohere_client = CohereClient(api_key=cohere_api_key)

    # OpenAI Embeddings setup
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Pinecone Index to store vectors to ( after user is logged in , this will be set to user's uuid )
    if "index_name" not in st.session_state:
        st.session_state.index_name = ""

    # session id ( for now , it's hardcoded value . later can be set to useruuid_unixtime format )
    if "session_id" not in st.session_state:
        st.session_state.session_id = "uniqueVALUE1234"

    # for disabling file uploader and submit button 
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False

    # key for file uploader widget , this increments by 1 whenever a pdf is uploaded
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    # Initialize store if not in session state
    if "store" not in st.session_state:
        st.session_state.store = {}

    ### Statefully manage chat history ###
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    selected_files = []

    # Display a temporary success message
    def temporary_success_message(message, duration=2):
        # Show success message
        success = st.success(message)
        # Wait for specified duration
        time.sleep(duration)
        # Clear the success message
        success.empty()

    # get_session_history function , to be used with RunnableWithMessageHistory class , this is used to pass session history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]    

    # Loading and Parsing Data with the help of LlamaParse
    def load_or_parse_data(user_folder, file_path, file_name):
        """
        Load or parse PDF data into markdown format.
        
        Args:
            user_folder (str): User's folder path
            file_path (str): Full path to the PDF file
            file_name (str): Name of the file
        
        Returns:
            parsed data or None if parsing fails
        """
        try:
            # Extract images and parse markdown
            images, parsed_data = extract_images_from_pdf(file_path, user_folder)
            
            # print(user_folder,file_path)
            updated_md_text = parsed_data.replace(user_folder, './')
            
            if updated_md_text is None:
                st.error("Failed to parse PDF")
                return None
            
            # Create markdown file with images and text
            markdown_path = os.path.join(user_folder, f"{file_name}.md")
            
            with open(markdown_path, "w", encoding="utf-8") as f:
                # Write text content
                for doc in updated_md_text:
                    f.write(doc if isinstance(doc, str) else doc.text + "\n")
                
                # Add image references
                f.write("\n## Images\n")
                for img in images:
                    # Use relative path for markdown image reference
                    relative_img_path = os.path.relpath(img['path'], user_folder)
                    f.write(f"\n![{img['filename']}]({relative_img_path})\n")
            
            print(f"Markdown file created: {markdown_path}")
            print(f"Number of images extracted: {len(images)}")
            
            return updated_md_text
        
        except Exception as e:
            st.error(f"An error occurred while loading or parsing the data: {e}")
            return None
     
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
            all_docs = []
            
            # track document sources and page number 
            doc_counter = 0
            
            for file_path, file_name in zip(file_paths, selected_files):
                # Call the function to either load or parse the data
                llama_parse_documents = load_or_parse_data(user_folder, file_path, file_name)
                if llama_parse_documents is None:
                    return
                
                images = extract_images_from_pdf(file_path, user_folder)
                # tables = extract_tables_from_pdf(file_path)
                
                # Convert to markdown
                markdown_path = os.path.join(user_folder, f"{file_name}.md")
                print("markdown_path", markdown_path)

                with open(markdown_path, "w", encoding="utf-8") as f:
                    for doc in llama_parse_documents:
                        f.write(doc if isinstance(doc, str) else doc.text + "\n")
                        # f.write(doc.text + "\n")

                loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")
                documents = loader.load()
                # documents = [Document(page_content=doc, metadata={"source": file_name, "page_number": i}) for i, doc in enumerate(llama_parse_documents)]
                
                # Enhance documents with sourse metadata
                for doc in documents:
                    related_images = [json.dumps(img)[:500] for img in images]  # Truncate large image metadata
                    doc.metadata.update({
                        "file_name":file_name,
                        "file_path": file_path,
                        "doc_id":f"doc_{doc_counter}",
                        "sourse_type": "pdf",
                        "create_timestamp": datetime.now().isoformat(),
                        "chunk_index": doc_counter,
                        "related_images": related_images  # Attach extracted image metadata
                    })
                    doc_counter +=1
                
                all_docs.extend(documents)

            # Split loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
            docs = text_splitter.split_documents(all_docs)
            
            # prepare texts and enhansed metadatas
            texts = []
            metadatas = []
            
            for i,doc in enumerate(docs):
                text = doc.page_content
                metadata = doc.metadata.copy()
            
            # Add chunk-specific metadata
                metadata.update({
                    "chunk_id": f"chunk_{i}",
                    "chunk_length": len(text),
                    "chunk_position": i,
                    "total_chunks": len(docs),
                    "processing_timestamp": datetime.now().isoformat()
                })
                # Extract and add any potential section headers or titles
                lines = text.split('\n')
                if lines and lines[0].strip():
                    metadata["chunk_title"] = lines[0].strip()[:100]
                
                texts.append(text)
                metadatas.append(metadata)
            
            print("texts",texts)
            print("metadatas",metadatas)
            print("st.session_state.index_name", st.session_state.index_name)
            
            PineconeVectorStore.from_texts(
                texts, embeddings, index_name=st.session_state.index_name, metadatas=metadatas
            )

            print("Vector DB created successfully!")
            return
        except Exception as e:
            print(f"Error details: {str(e)}")
            st.error(f"An error occurred while creating the vector database: {e}")

    def process_selected_files(save_folder, email):
        try:
            file_paths = []
            for file in selected_files:
                file_path = os.path.join(save_folder, file)
                file_paths.append(file_path)
            print("file paths :",file_paths)
            
            # Check if the index exists
            existing_indexes = pc.list_indexes()
            print("existing_indexes list :",existing_indexes)
            print("index_name to find :",st.session_state.index_name)

            if any(index.name == st.session_state.index_name for index in existing_indexes):
                # Delete the existing index
                pc.delete_index(st.session_state.index_name)
                print(f"Deleted existing index: ",{st.session_state.index_name})

            print("creating new index")
            # Create a new index with the same name
            pc.create_index(
                name=st.session_state.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"Created new index: ", {st.session_state.index_name})
            # Create user-specific directory in data/
            user_folder = os.path.join("data", email)
            os.makedirs(user_folder, exist_ok=True)
            
            # Create the vector database for multiple files
            create_vector_database(user_folder, file_paths, selected_files)
            
            # Save the names of the files that were converted
            selected_file_folder = os.path.join("selected", email)
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


    def disable():
        st.session_state.disabled = True
        
    def disableOff():
        st.session_state.disabled = False

    def save_file(save_folder,file):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        file_path = os.path.join(save_folder, file.name)
        with open(file_path, mode='wb') as w:
            w.write(file.getvalue())
            st.session_state["file_uploader_key"] += 1
            # st.sidebar.success(f"File {file.name} uploaded successfully!")

    # # generate response 
    # def generate_response(prompt: str) :
    #     try:
    #         contextualize_q_system_prompt = (
    #             "Given a chat history and the latest user question "
    #             "which might reference context in the chat history, "
    #             "formulate a standalone question which can be understood "
    #             "without the chat history. Do NOT answer the question, "
    #             "just reformulate it if needed and otherwise return it as is."
    #         )
                
    #         contextualize_q_prompt = ChatPromptTemplate.from_messages(
    #                 [
    #                     ("system", contextualize_q_system_prompt),
    #                     MessagesPlaceholder("chat_history"),
    #                     ("human", "{input}"),
    #                 ]
    #         )

    #         # Reranker 
    #         def reRanker():
    #             compressor = CohereRerank(model="rerank-english-v3.0",client=cohere_client)
    #             vectorStore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
    #             compression_retriever = ContextualCompressionRetriever(
    #                 base_compressor=compressor,
    #                 base_retriever=vectorStore.as_retriever(
    #                     search_kwargs={"k": 5},
    #                 ),
    #             )
    #             return compression_retriever

    #         compression_retriever = reRanker()

    #         history_aware_retriever = create_history_aware_retriever(
    #                 llm, compression_retriever, contextualize_q_prompt
    #         )

    #         system_prompt = """You are a specialized document analysis assistant designed to provide precise answers by synthesizing information from tables, structured text, and visual elements within provided PDF documents, with advanced capabilities for mathematical calculations and reasoning.
    #         When responding to questions:

    #         1. **Table Data Analysis:**
    #         - Extract exact numerical values and relationships from tables, maintaining the structure
    #         - ALWAYS Format table data to display in a tabular layout
    #         - Specify table titles and numbers to clearly identify sources
    #         - Include any footnotes or special notations, ensuring data context remains intact
    #         - For mathematical operations on table data:
    #             * First display the relevant table data being used
    #             * Show each mathematical step separately with clear labels
    #             * Include subtotals for complex calculations
    #             * Validate results by cross-checking across different tables if applicable

    #         2. **Mathematical Reasoning and Calculations:**
    #         - For any calculation, follow these steps:
    #             1. Clearly state the mathematical problem to be solved
    #             2. List all relevant values and their sources (page numbers, table numbers)
    #             3. Show each calculation step with explanations
    #             4. Use proper mathematical notation and units
    #             5. Provide intermediate results for complex calculations
    #             6. Double-check calculations and show verification steps
    #             7. Present the final result with appropriate context
    #         - When performing calculations across multiple tables:
    #             * First organize all relevant data in a structured format
    #             * Show relationships between different data sources
    #             * Explain any assumptions or data transformations
    #             * Validate consistency of units and formats before calculations

    #         3. **Visual Content Interpretation:**
    #         - Describe data shown in charts and graphs with details on values, trends, and patterns
    #         - Reference relevant axis labels, legends, and scales
    #         - Extract numerical data from graphs for calculations when needed
    #         - Show mathematical relationships between visual data points
    #         - Summarize findings with direct connections to related text for holistic insight

    #         4. **Textual Information:**
    #         - Cite sections and page numbers when quoting or referencing text
    #         - Organize text data with original formatting, including bullet points, numbered lists, and paragraph structures
    #         - Note any footnotes or cross-references, ensuring information is captured in its hierarchical order
    #         - Extract numerical information from text for calculations when relevant

    #         **Response Format Guidelines:**
    #         - Identify source elements (table, text, or visual) at the beginning of responses
    #         - For tables: Display data in a table format for readability
    #         - For calculations:
    #         * Use markdown code blocks for showing calculation steps
    #         * Format mathematical equations clearly
    #         * Include units in each step
    #         * Show intermediate results
    #         - For text: Retain PDF-style formatting, using bullets or lists as found in the document
    #         - For visuals: Summarize visual data with references to axes and legends
    #         - Include precise locations (page numbers, section numbers) for all referenced data
    #         - Cross-reference across document elements when relevant, showing interconnections
    #         - Maintain original data precision, units, and context qualifiers

    #         **For Mathematical Operations:**
    #         ```
    #         Step 1: State the calculation objective
    #         Step 2: List source data with references
    #         Step 3: Show calculation setup
    #         Step 4: Perform operations step by step
    #         Step 5: Verify results
    #         Step 6: Present final answer with context
    #         ```

    #         If the required information cannot be found in the provided PDF content, respond with: "I cannot locate specific information about this in the provided PDF documents. Please verify if this information is present in the documents or rephrase your question."

    #         You may respond to basic greetings, but for all other queries, strictly use information from the provided documents.

    #         {context}"""


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

    def _register_credentials(email: str, name: str, password: str):
        if not validator.validate_name(name):
            st.error('Name is not valid')
        if not validator.validate_email(email):
            st.error('Email is not valid')
        try:
            collection.insert_one( {'password': Hasher([password]).generate()[0],'email':email,'name':name } )
        except Exception as e :
            st.error(e)

    # App Title / App Name
    st.title('DocuMindz :bookmark_tabs:')

    if st.session_state["authentication_status"] is None or st.session_state["authentication_status"] is False:
        menu = ["Login","Register"]
        # Sidebar Image
        st.sidebar.image('images/logo.png')
        choice = st.sidebar.selectbox("Menu",menu)
        if choice == "Login":
            authenticator.login('Login', 'main')
        elif choice == "Register":
                register_user_form = st.form('Register user')
                register_user_form.subheader("Register")
                new_email = register_user_form.text_input('Email')
                new_name = register_user_form.text_input('Name')
                new_password = register_user_form.text_input('Password', type='password')
                new_password_repeat = register_user_form.text_input('Repeat password', type='password')
                if register_user_form.form_submit_button('Register'):
                    user_document = collection.find_one({"email": new_email})
                    if len(new_email)  and len(new_name) and len(new_password) > 0:
                        if not user_document:
                            if new_password == new_password_repeat:         
                                _register_credentials(new_email, new_name, new_password)
                                st.success('User registered successfully')
                            else:
                                st.error('Passwords do not match')
                        else:
                            st.error('Email already taken')
                    else:
                        st.error('Please enter an email, name, and password')

    if st.session_state["authentication_status"]:
        print("session_state after authentication_status is true :",st.session_state)
        email = st.session_state["email"]

        # PDF files directory (to save PDF files to local db)
        save_folder = f"PDF_PATH/{email}"
        print("save folder :",save_folder)

        userData = collection.find_one({"email":email})
        print("user id by email :",userData["_id"])
        userId = userData["_id"]
        st.session_state.index_name = str(userId)
        print("user's unique id :",userId)
        print("st.session_state.index_name is set to userId :",st.session_state.index_name)

        # Sidebar Image
        st.sidebar.image('images/logo.png')
        # File Uploader Widget ( as form ) in Streamlit Sidebar
        # st.sidebar.title('File Upload and Processing')

        with st.sidebar.form(key='sidebar_form'):
            # Allow the user to upload a file
            uploaded_files = st.file_uploader("Upload a file", type=["pdf"], key=st.session_state["file_uploader_key"], disabled=st.session_state.disabled, accept_multiple_files=True)
            # If a file was uploaded, display its contents
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_file(save_folder,uploaded_file)
                st.success(f'files uploaded successfully')

            submit_btn = st.form_submit_button('Upload',
                                                on_click=disable,
                                                disabled=st.session_state.disabled)
            if submit_btn:
                if uploaded_files is None:
                    st.error("Select a file first !!!")
                    disableOff()
                    st.rerun()
                else:
                    disableOff()
                    st.rerun()

        # Function to list files in a directory and check if they are in the selected files list
        def list_files_in_directory(directory, selected_file_path):
            try:
                if os.path.exists(directory):
                    files = os.listdir(directory)
                    saved_selected_files = []
                    if os.path.exists(selected_file_path):
                        with open(selected_file_path, "r") as f:
                            saved_selected_files = f.read().splitlines()
                    return files, saved_selected_files
                else:
                    return [], []
            except OSError as e:
                print(f"An error occurred while accessing the directory: {e}")
                return [], []
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return [], []

        
        # Display the list of uploaded files with delete buttons
        st.sidebar.write("### Uploaded Files:")
        selected_file_path = f"selected/{email}/selected.txt"

        uploaded_files_list, saved_selected_files = list_files_in_directory(save_folder, selected_file_path)

        # Function to delete a file
        def delete_file(file_path, selected_file_path, email, file):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    # st.sidebar.success(f"File {file} deleted successfully!")
                    
                    # Check if the file is in the selected files list
                    if os.path.exists(selected_file_path):
                        with open(selected_file_path, "r") as f:
                            selected_files = f.read().splitlines()

                        if file in selected_files:
                            # Remove the file from the selected files list
                            selected_files.remove(file)
                            with open(selected_file_path, "w") as f:
                                for selected_file in selected_files:
                                    f.write(selected_file + "\n")
                            
                            # Delete Pinecone index if no files are left in selected files
                            if not selected_files:
                                index_name = st.session_state.index_name
                                existing_indexes = pc.list_indexes()
                                if any(index.name == index_name for index in existing_indexes):
                                    pc.delete_index(index_name)
                                    st.sidebar.success(f"Pinecone index for {file} deleted successfully!")
            except Exception as e:
                st.sidebar.error(f"An error occurred while deleting the file: {e}")


        for file in uploaded_files_list:
            try:
                file_path = os.path.join(save_folder, file)
                col1, col2 = st.sidebar.columns([3, 1])
                # Pre-fill the checkbox if the file is in the selected files list
                checkbox = col1.checkbox(file, key=f"checkbox_{file}", value=(file in saved_selected_files))
                if checkbox:
                    selected_files.append(file)
                if col2.button("❌", key=f"delete_{file}"):
                    delete_file(file_path, selected_file_path, email, file)
                    st.rerun()  # Refresh the app to update the file list
            except Exception as e:
                st.sidebar.error(f"An error occurred while rendering the file list: {e}")

        if len(uploaded_files_list)>0:
            processBtn = st.sidebar.button("Process Selected Files",disabled=len(selected_files)==0)
            if processBtn:
                # Process the selected files
                with st.spinner("Processing files..."):
                    process_selected_files(save_folder, email)
                    temporary_success_message("Files Processed Successfully")


        authenticator.logout('Logout', 'sidebar','logout-key')

        # Conversation History
        for message in st.session_state.chat_history:
            if isinstance(message,HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                with st.chat_message("AI"):
                    st.markdown(message.content)

        prompt = st.chat_input("Hey, What's up?")

        if prompt is not None and prompt !="" :
            existing_indexes = pc.list_indexes()
            if any(index.name == st.session_state.index_name for index in existing_indexes): 
                st.session_state.chat_history.append(HumanMessage(prompt))
                with st.chat_message("Human"):
                    st.markdown(prompt)

                if len(pc.list_indexes()) == 0:
                    st.error("Please upload some files first!")
                else:
                    with st.chat_message("AI"):
                        ai_response =generate_response(prompt,st.session_state.index_name,st.session_state.chat_history)
                        st.write(ai_response)

                    st.session_state.chat_history.append(AIMessage(ai_response))
            else:
                st.error("Upload a PDF and process it first !!!")

    elif st.session_state["authentication_status"] is False:
        st.error('Email/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     st.warning('Please enter your email and password')

    # print("file ran last")

if __name__ == "__main__":
    main()