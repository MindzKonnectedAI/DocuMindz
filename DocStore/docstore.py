from langchain_community.storage import MongoDBStore
import os
import streamlit as st
from Utils.session_states import initialize_session_states

# Initialize session states
initialize_session_states()

MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")

mongo_docstore = MongoDBStore(MONGO_DB_CONN_STR, db_name="new",collection_name=st.session_state.index_name)