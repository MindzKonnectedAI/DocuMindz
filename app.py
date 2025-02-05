#  UI components
import streamlit as st # streamlit module ( for building UI )
import os # Operating System module
from dotenv import load_dotenv  # .env file loading

#  LLM and Chat Components
from langchain_core.messages import HumanMessage, AIMessage

from pinecone import ServerlessSpec  # Pinecone as Vector DB (Pinecone's Python Library)

# Authentication & Database
import streamlit_authenticator_mongo as stauth
from Database.DocumentDatabase.dbscript import collection,get_chat_history
from streamlit_authenticator_mongo.validator import Validator
from streamlit_authenticator_mongo.hasher import Hasher

from langchain.globals import set_verbose
import nest_asyncio # not sure if this was needed in PDFRAG app
import yaml
from yaml.loader import SafeLoader
import uuid

# Response Generator function
from ResponseGenerator.llmresponse import generate_response

from Utils.utilities import temporary_success_message,save_file,disable,disableOff,list_files_in_directory,delete_file,update_dossier,get_default_index

from Database.VectorDatabase.pinecone import pc,process_selected_files,list_existing_indexes,wait_on_index,wait_on_namespace
from Utils.session_states import initialize_session_states

def main():
    # defaults
    load_dotenv(override=True)
    nest_asyncio.apply()
    set_verbose(True) 
    # Initialize session states
    initialize_session_states()

    # Page Configuration
    st.set_page_config("DocuMindz",":bookmark_tabs:")

    # App Title / App Name
    st.title('DocuMindz :bookmark_tabs:')
    st.subheader("Simplify Documents, Amplify Decisions")

    # stauth package's validator
    validator = Validator()

    # config file of stauth package
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    # authenticator setup
    authenticator = stauth.Authenticate(
        collection,
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )        

    def _register_credentials(email: str, name: str, password: str):
        if not validator.validate_name(name):
            st.error('Name is not valid')
        if not validator.validate_email(email):
            st.error('Email is not valid')
        try:
            collection.insert_one( {'password': Hasher([password]).generate()[0],'email':email,'name':name } )
        except Exception as e :
            st.error(e)

    selected_files = []

    @st.dialog("Create Dossier", width="large")
    def create_dossier():
        try:
            with st.form(key="create_dossier_key"):
                dossier_name = st.text_input("Dossier Name")
                form_submitted = st.form_submit_button(label="Submit")
                if form_submitted:
                    with st.spinner(f'Creating Dossier "{dossier_name}"'):
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

                        # Create a new dossier
                        index = pc.Index(st.session_state.index_name)
                        # Upsert a dummy vector to create the namespace
                        index.upsert(
                            vectors=[
                                {"id": "dummy", "values": [0.1] * 3072,"metadata":{"doc_id":str(uuid.uuid4()),"text":"this is dummy text"}}
                            ],
                            namespace=dossier_name,
                        )
                        # Wait until the namespace is created
                        wait_on_namespace(st.session_state.index_name, dossier_name)
                        st.rerun()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


    if st.session_state["authentication_status"] is None or st.session_state["authentication_status"] is False:
        menu = ["Login","Register"]
        # Sidebar Image
        # st.sidebar.image('images/company-logo.png',width=200)
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

    if st.session_state["authentication_status"] is True:
        print("session_state after authentication_status is true :",st.session_state)
        email = st.session_state["email"]

        # PDF files directory (to save PDF files to local db)
        print("st.session_state.namespace :",st.session_state.namespace)
        save_folder = f"PDF_PATH/{email}/{st.session_state.namespace}"
        print("save folder :",save_folder)
        selected_file_path = f"selected/{email}/{st.session_state.namespace}/selected.txt"
        print("selected_file_path :",selected_file_path)
        userData = collection.find_one({"email":email})
        print("user id by email :",userData["_id"])
        userId = userData["_id"]
        st.session_state.index_name = str(userId)
        print("type of user's unique id :",type(userId))
        print("st.session_state.index_name is set to userId :",st.session_state.index_name)

        # Sidebar Image
        # st.sidebar.image('images/company-logo.png',width=200)

        with st.sidebar.form(key='sidebar_form'):
            # Allow the user to upload a file
            uploaded_files = st.file_uploader("Select documents", type=["pdf"], key=st.session_state["file_uploader_key"], disabled=st.session_state.disabled, accept_multiple_files=True)
            # If a file was uploaded, display its contents
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_file(save_folder,uploaded_file)
                st.success(f'files uploaded successfully')

            submit_btn = st.form_submit_button('Upload',on_click=disable,disabled=st.session_state.disabled)
            if submit_btn:
                if uploaded_files is None:
                    st.error("Select a file first !!!")
                    disableOff()
                    st.rerun()
                else:
                    disableOff()
                    st.rerun()
        
        dossierList = ["Default"]
        st.sidebar.write("### Dossiers:")

        
        try:
            index = pc.describe_index(st.session_state.index_name)
            described_index = pc.Index(host=index.host)
            index_stats = described_index.describe_index_stats()
            print("index_stats :",index_stats)
                
            # Extracting namespace names into a list and replacing '' with 'Default'
            namespace_names = [
                'Default' if name == '' else name for name in index_stats['namespaces'].keys()
            ]

            print("namespace_names list :",namespace_names)
            combined_dossiers = list(dict.fromkeys(dossierList + namespace_names))
            st.sidebar.radio(
                "Select Dossier to Chat",
                combined_dossiers,
                key="dossier_radio",
                label_visibility="collapsed",
                on_change=update_dossier,
                index=get_default_index(combined_dossiers)
            )

        except Exception as e:
            st.sidebar.radio(
                "Select Dossier to Chat",
                dossierList,
                key="dossier_radio",
                label_visibility="collapsed",
                on_change=update_dossier,
                index=get_default_index(dossierList)
            )
            

        st.sidebar.button("Create Dossier",key=uuid.uuid4(),on_click=create_dossier)

        # Display the list of uploaded files with delete buttons
        st.sidebar.write("### Uploaded Files:")

        uploaded_files_list, saved_selected_files = list_files_in_directory(save_folder, selected_file_path)

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
                    process_selected_files(save_folder, email,selected_files)
                    temporary_success_message("Files Processed Successfully")

        authenticator.logout('Logout', 'sidebar','logout-key')

        chat_history = get_chat_history(st.session_state.namespace+"_"+st.session_state.session_id)

        # Conversation History
        for message in chat_history:
            if isinstance(message,HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                with st.chat_message("AI"):
                    st.markdown(message.content)

        prompt = st.chat_input("Hey, What's up?")

        if prompt is not None and prompt !="" :
            existing_indexes = list_existing_indexes()
            if any(index.name == st.session_state.index_name for index in existing_indexes): 
                with st.chat_message("Human"):
                    st.markdown(prompt)

                if len(existing_indexes) == 0:
                    st.error("Please upload some files first!")
                else:
                    with st.chat_message("AI"):
                        ai_response = generate_response(prompt)
                        st.markdown(ai_response)
                        # ai_response = st.write_stream(generate_response(prompt))
                    # st.session_state.chat_history.append(AIMessage(ai_response))
            else:
                st.error("Upload a PDF and process it first !!!")

    elif st.session_state["authentication_status"] is False:
        st.error('Email/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     st.warning('Please enter your email and password')

    # print("file ran last")

if __name__ == "__main__":
    main()