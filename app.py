#  UI components
import io
from langchain import hub
import streamlit as st # streamlit module ( for building UI )
import os # Operating System module
from dotenv import load_dotenv  # .env file loading


#  LLM and Chat Components
from langchain_openai import ChatOpenAI, OpenAI # OpenAI's Chat Model
from langchain_core.messages import HumanMessage, AIMessage

# PDF Processing 
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text splitter ( TextSplitter to split Markdown )
from langchain_experimental.text_splitter import SemanticChunker

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

# Chat History  Mangements
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Authentication & Database
import streamlit_authenticator_mongo as stauth
from dbscript import collection
from streamlit_authenticator_mongo.validator import Validator
from streamlit_authenticator_mongo.hasher import Hasher

from langchain.globals import set_verbose
import nest_asyncio # not sure if this was needed in PDFRAG app
import yaml
from yaml.loader import SafeLoader

import pymupdf4llm
import time
import base64
from PIL import Image
import fitz
from langchain_core.output_parsers import StrOutputParser

from openai import Client
import uuid
# from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore  # Langchain's Pinecone library
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

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

    #New addition 1
    def extract_images_and_text(file_path):
        doc = fitz.open(file_path)
        images_and_text = []

        for page in doc:
            text = page.get_text()
            # print("text :",text)
            image_list = page.get_images(full=True)
            # print("image_list :",image_list)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                image = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                # words = text.split()
                # text_before_image = " ".join(words[-200:]) if len(words) > 200 else text

                # page text becomes the image context
                text_before_image = text

                images_and_text.append({
                  "xref":xref,
                  "base64_image": img_base64,
                  "text": text_before_image
            })
            # print("xref :",xref)
            # print("image ",img_base64)
            # print("text ",text_before_image)
            # print("-"*50)


    
        return images_and_text #as it is
    
 
    # Create a new array with unique xrefs
    def get_unique_xref_images_and_text(pdf_image_and_text_array):
        unique_xref_dict = {}
        unique_xref_array = []

        for entry in pdf_image_and_text_array:
          xref = entry["xref"]
          if xref not in unique_xref_dict:
            unique_xref_dict[xref] = entry
            unique_xref_array.append(entry) #as it is

        return unique_xref_array
    
    # Do not Disturb
    def extract_images_from_pdf(file_path, user_folder):
        """ 
        Extract images from PDF with enhanced metadata using pymupdf4llm.
        Args:
        file_path (str): Path to the PDF file.
        user_folder (str): Path to the user's folder to save extracted images.
        
        Returns:
        list: List of dictionaries containing image details.
        """
        
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
            pdf_image_and_text_array= extract_images_and_text(file_path) #->pdf_image_and_text_array 
            unique_xref_array= get_unique_xref_images_and_text(pdf_image_and_text_array) #-> return unique_xref_array


            return parsed_data, unique_xref_array
        #return parsed_data, unique_xref_array
        
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
            
            return [], None
        
    def convert_image_array_to_documents(unique_xref_array):
        """
        Converts an array of image objects to LangChain Document instances.
        
        Args:
            image_array: List of dictionaries containing 'xref', 'base64_image', and 'text'.

        Returns:
            List of LangChain Document instances.
        """
        documents = []
        for item in unique_xref_array:
            try:
                # Extract fields from the item
                page_content = item.get("base64_image", "")
                metadata = {
                    "xref": item.get("xref", ""),
                    "surrounding_text": item.get("text", "")
                }
                # Create a Document instance
                document = Document(page_content=page_content, metadata=metadata)
                documents.append(document)
            except Exception as e:
                print(f"Error converting item to Document: {e}")
        return documents

        
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

    # Initialize chunking_strategy if not in session state
    if "chunking_strategy" not in st.session_state:
        st.session_state.chunking_strategy = "Semantic"

    ### Statefully manage chat history ###
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "doc_store" not in st.session_state:
        st.session_state.doc_store = InMemoryStore()

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
    
    def create_text_summaries(docs):
        print('inside text_summaries')
        # print ('Docs', docs)
        prompt_text = """
          You are an assistant tasked with summarizing tables and text.
          Give a concise summary of the table or text.

          Respond only with the summary, no additionnal comment.
          Do not start your message by saying "Here is a summary" or anything like that.
          Just give the summary as it is.

          Table or text chunk: {element} 

                     """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Summary chain
        model = ChatOpenAI(
        temperature=0, model_name="gpt-4o-mini")
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        # Extract content from Document objects
        text_chunks = [doc.page_content for doc in docs]

        # Summarize text
        text_summaries = summarize_chain.batch(text_chunks, {"max_concurrency": 3}) #docs
        return text_summaries
    
    # def create_image_summaries(image_docs):
    #     print('inside image_summaries')
    #     print('Image Docs:', image_docs)
        
    #     # Template for generating image captions
    #     prompt_image = """
    #         You are a highly advanced multimodal assistant specializing in generating concise and accurate textual summaries for images.
    #         Your role is to analyze the visual content of images, including objects, scenes, actions, and emotions, and describe them in a way that captures the essence of the image.

    #         Each summary should:
    #         1. Be no longer than 2-3 sentences.
    #         2. Focus on key elements in the image (e.g., objects, settings, interactions, and emotions).
    #         3. Avoid unnecessary details or speculative information.
    #         4. Use formal and neutral language suitable for embedding generation.

    #         Image data: {base64_image}
    #         Context text: {text}
    #     """
    #     prompt = ChatPromptTemplate.from_template(prompt_image)

    #     # Summary chain
    #     model = ChatOpenAI(
    #         temperature=0.5, model_name="gpt-4o-mini")
    #     summarize_chain = {
    #         "base64_image": lambda x: x["base64_image"],
    #         "text": lambda x: x["text"],
    #     } | prompt | model | StrOutputParser()

    #     # Summarize images
    #     # image_summaries = summarize_chain.batch(image_docs)
    #     # print('image_summaries:', image_summaries)
    #     # return image_summaries
    #     image_summaries=[]
    #     batch_size = 5  # Adjust batch size to fit within token limits
    #     for i in range(0, len(image_docs), batch_size):
    #         batch = image_docs[i:i + batch_size]
    #         summaries = summarize_chain.batch(batch, {"max_concurrency": 2})  # Reduce concurrency if needed
    #         image_summaries.extend(summaries)

    def generate_caption_for_image(base_img, query):
          messages = []
          system_role = {
                "role": "system",
                "content": '''You are a highly advanced multimodal assistant specializing in generating concise and accurate textual summaries for images. Your role is to analyze the visual content of images, including objects, scenes, actions, and emotions, and describe them in a way that captures the essence of the image.

                Each summary should:
                1. Be no longer than 2-3 sentences.
                2. Focus on key elements in the image (e.g., objects, settings, interactions, and emotions).
                3. Avoid unnecessary details or speculative information.
                4. Use formal and neutral language suitable for embedding generation.

                Your output will be used to generate embeddings for semantic search and vector storage, so ensure the summaries are informative and contextually rich.'''
            }
          messages.append(system_role)
          messages.append({"role": "user", "content": f"this is the text... {query}"})
          client = Client(api_key=os.getenv("OPENAI_API_KEY"))
        #   print ('messages', messages)
          response = client.chat.completions.create(
             model="gpt-4o-mini",
             messages=[*messages,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base_img}",
                            "detail": "high"
                           }
                       }
                   ]
               }
           ],
           temperature=0.5,
           top_p=0.9,
           presence_penalty=0.6,
           frequency_penalty=0.5
    )
          assistant_message = response.choices[0].message.content
          messages.append({"role": "assistant", "content": assistant_message})
          return assistant_message       

    def create_image_summaries(unique_xref_array):
        print('inside image_summaries')
        image_summaries = []
        for item in unique_xref_array:
           caption = generate_caption_for_image(item['base64_image'], item['text'])
           image_summaries.append(caption)
        return image_summaries

    def map_image_keys(array_of_dicts):
        """
        Maps an array of dictionaries to an array of strings containing the values of the 'image' key.

        Parameters:
            array_of_dicts (list): A list of dictionaries.

        Returns:
            list: A list of strings corresponding to the values of the 'image' key in each dictionary.
        """
        return [d.get('base64_image', '') for d in array_of_dicts]

    
    
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
                
                parsed_data, unique_xref_array = extract_images_from_pdf(file_path, user_folder)
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
                docs = [Document(page_content=text, metadata={"source": file_name}) for text in chunked_texts]

                # print ('Docs', docs)
                text_summaries= create_text_summaries(docs) #-> return text summaries 
                print ('length of text summary', len(text_summaries)) 
                # print ('this is text summary',text_summaries) 

                image_summaries= create_image_summaries(unique_xref_array) #-> return image summaries
                print ('length of image summary', len(image_summaries))
                # print ('this is image summary', image_summaries)

                # Pinecone setup (for vector storage)
                
                vectorstore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
                # The storage layer for the parent documents
                # store = InMemoryStore()
                id_key = "doc_id"

                # The retriever (empty to start)
                retriever = MultiVectorRetriever(
                    vectorstore=vectorstore,
                    docstore=st.session_state.doc_store,
                    id_key=id_key,
                )
                # Add texts
                doc_ids = [str(uuid.uuid4()) for _ in docs]
                summary_texts = [
                    Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
                ]
                retriever.vectorstore.add_documents(summary_texts)
                retriever.docstore.mset(list(zip(doc_ids, docs)))

                # final_array = map_image_keys(unique_xref_array)
                # print(final_array)  # Output: ['image1.jpg', 'image2.png', 'image3.gif']
                
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

    # generate response 
    def generate_response(prompt: str) :
        try:
            contextualize_q_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

            # Reranker 
            def reRanker():
                compressor = CohereRerank(model="rerank-english-v3.0",client=cohere_client)
                vectorStore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
                id_key = "doc_id"

                retriever = MultiVectorRetriever(
                    vectorstore=vectorStore,
                    docstore=st.session_state.doc_store,
                    id_key=id_key,
                )

                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever,
                )
                return compression_retriever

            compression_retriever = reRanker()

            history_aware_retriever = create_history_aware_retriever(
                    llm, compression_retriever, contextualize_q_prompt
            )

            system_prompt = """You are a specialized document analysis assistant designed to provide precise, context-rich answers by synthesizing information from tables, 
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

            {context}"""


            chatPrompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
            )
                            
            question_answer_chain = create_stuff_documents_chain(llm, chatPrompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    output_messages_key="answer",
                    history_messages_key="chat_history",
            )
            for chunk in conversational_rag_chain.stream(input={"input": prompt},config={'configurable': {'session_id': st.session_state.session_id}}):
                answer_chunk = chunk.get("answer")
                if answer_chunk:
                    yield answer_chunk
        except Exception as e:
            st.error(f"An error occurred while generating the response: {e}")

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

        st.session_state.chunking_strategy = st.sidebar.radio(
            "Select Document Chunking Strategy",
            ["Semantic","Recursive"],
        )
        print("st.session_state.chunking_strategy :",st.session_state.chunking_strategy)

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
                        ai_response = st.write_stream(generate_response(prompt))

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