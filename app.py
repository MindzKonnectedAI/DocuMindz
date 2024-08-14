import streamlit as st # streamlit module ( for building UI )
import os # Operating System module
from langchain_openai import ChatOpenAI # OpenAI's Chat Model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv  # .env file loading
from llama_parse import LlamaParse #  LlamaParse for parsing PDF files
from langchain_community.document_loaders import UnstructuredMarkdownLoader # Markdown file loader ( because we're using LlamaParse )
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text splitter ( TextSplitter to split Markdown )
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec  # Pinecone as Vector DB (Pinecone's Python Library)
from langchain_pinecone import PineconeVectorStore  # Langchain's Pinecone library
from langchain.retrievers import ContextualCompressionRetriever
from cohere.client import Client as CohereClient
from langchain_cohere import CohereRerank  # CohereRerank for reranking
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.globals import set_verbose
import joblib
import nest_asyncio # not sure if this was needed in PDFRAG app
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator_mongo as stauth
from dbscript import collection
from streamlit_authenticator_mongo.validator import Validator
from streamlit_authenticator_mongo.hasher import Hasher


def main():

    # Page Configuration
    st.set_page_config("DocuMindz",":bookmark_tabs:")

    # stauth package's validator
    validator = Validator()

    # config file of stauth package
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    # defaults
    load_dotenv()
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

    # LlamaParse api key
    llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
    parsingInstructionUber10k = """The provided document is unstructured
    It contains many tables, text, image and list.
    Try to be precise while answering the questions"""
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",  # we want md file back
        parsing_instruction=parsingInstructionUber10k,
        max_timeout=5000,
    )

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

    # Cohere model API key and configuration
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

    # get_session_history function , to be used with RunnableWithMessageHistory class , this is used to pass session history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    # Creating pkl string (required for llamaParser to work efficiently)
    def create_pkl_string(filename):
        file_name, extension = os.path.splitext(filename)
        new_string = file_name + ".pkl"
        return new_string

    # Loading and Parsing Data with the help of LlamaParse
    def load_or_parse_data(user_folder,file_path,file_name):
        # LlamaParse creates a pkl file
        # PDF -> pkl -> md -> vector
        try:
            changed_file_ext = create_pkl_string(file_name)
            data_file = os.path.join(user_folder, changed_file_ext)

            if os.path.exists(data_file):
                # Load the parsed data from the file
                parsed_data = joblib.load(data_file)
            else:
                # Perform the parsing step and store the result in llama_parse_documents
                llama_parse_documents = parser.load_data(file_path)
                # Save the parsed data to a file
                print("Saving the parse results in .pkl format ..........")
                joblib.dump(llama_parse_documents, data_file)

                # Set the parsed data to the variable
                parsed_data = llama_parse_documents

            return parsed_data
        except Exception as e:
            st.error(f"An error occurred while loading or parsing the data: {e}")
            return None
        
        # Create vector database for multiple files
    def create_vector_database(user_folder, file_paths):
        """
        Creates a vector database using document loaders and embeddings for multiple files.

        This function loads PDF documents,
        splits the loaded documents into chunks, transforms them into embeddings using OpenAIEmbeddings,
        and finally persists the embeddings into a Pinecone vector database.
        """
        try:
            print("Inside create_vector_database function")
            all_docs = []
            for file_path, file_name in zip(file_paths, selected_files):
                # Call the function to either load or parse the data
                llama_parse_documents = load_or_parse_data(user_folder, file_path, file_name)
                if llama_parse_documents is None:
                    return

                markdown_path = os.path.join(user_folder, f"{file_name}.md")
                print("markdown_path", markdown_path)

                with open(markdown_path, "w", encoding="utf-8") as f:
                    for doc in llama_parse_documents:
                        f.write(doc.text + "\n")

                loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")
                documents = loader.load()
                all_docs.extend(documents)

            # Split loaded documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(all_docs)

            # Prepare texts and metadatas
            texts = [d.page_content for d in docs]
            print("texts :",texts)
            metadatas = [d.metadata for d in docs]
            print("metadatas :",metadatas)
            print("st.session_state.index_name :",st.session_state.index_name)
            PineconeVectorStore.from_texts(
                texts, embeddings, index_name=st.session_state.index_name, metadatas=metadatas
            )

            print("Vector DB created successfully!")
            return
        except Exception as e:
            st.error(f"An error occurred while creating the vector database: {e}")

    def process_selected_files(save_folder, email):
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
        create_vector_database(user_folder, file_paths)
        # Save the names of the files that were converted
        selected_file_folder = os.path.join("selected", email)
        os.makedirs(selected_file_folder, exist_ok=True)
        text_file_path = os.path.join(selected_file_folder, "selected.txt")
        with open(text_file_path, "w") as f:
            for file_name in selected_files:
                f.write(file_name + "\n")


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
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
                
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
            )

            # Reranker 
            def reRanker():
                compressor = CohereRerank(model="rerank-english-v3.0",client=cohere_client)
                vectorStore = PineconeVectorStore(index_name=st.session_state.index_name, embedding=embeddings)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vectorStore.as_retriever(
                        search_kwargs={"k": 5},
                    ),
                )
                return compression_retriever


            compression_retriever = reRanker()

            history_aware_retriever = create_history_aware_retriever(
                    llm, compression_retriever, contextualize_q_prompt
            )

            system_prompt = (
                    "You are an assistant designed to answer questions strictly based on the content of provided PDF documents. "
                    "You may respond to common greetings like 'Hi' or 'Hello' and summarize the content of the PDFs. "
                    "For all other questions, only use the information contained within the PDFs."
                    "If you cannot find the answer in the provided context, respond with: `I'm sorry, but I couldn't find information about that in the provided PDF documents.`"
                    "Do not use any external knowledge beyond the PDFs."
                    "\n\n"
                    "{context}"
            )

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

        # File Uploader Widget ( as form ) in Streamlit Sidebar
        st.sidebar.title('File Upload and Processing')

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

        authenticator.logout('Logout', 'sidebar')
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