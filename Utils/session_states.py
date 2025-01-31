import streamlit as st

def initialize_session_states():

    # Pinecone Index to store vectors to ( after user is logged in , this will be set to user's uuid )
    if "index_name" not in st.session_state:
        st.session_state.index_name = "null"
        
    # Pinecone Namespace to store document vectors
    if "namespace" not in st.session_state:
        st.session_state.namespace = "Default"

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
    
    # Initialize chat_history for the current dossier
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Default empty chat history
        
    ### Statefully manage chat history ###
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}  # Dictionary to store chat histories per dossier
        

    if st.session_state.namespace not in st.session_state.chat_histories:
        st.session_state.chat_histories[st.session_state.namespace] = []

    # Ensure chat_history always refers to the current dossier
    st.session_state.chat_history = st.session_state.chat_histories[st.session_state.namespace]

        
    
    