import pymongo
import os
from dotenv import load_dotenv  # .env file loading
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import streamlit as st

load_dotenv(override=True)
# User credentials will be saved in 'new' database

MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")

# Prerequisite : MongoDB Compass
# client = pymongo.MongoClient("localhost", 27017)

# In case you want to use MongoDB Atlas instead of MongoDB Compass
client = pymongo.MongoClient(MONGO_DB_CONN_STR)

# DocuMindz ( DB Name )
db = client.new 

# Users collection ( Collection Name)
collection = db.new

def get_chat_history(session_id: str) -> MongoDBChatMessageHistory:
    mongo_db_chat_message_history= MongoDBChatMessageHistory(
        MONGO_DB_CONN_STR , session_id, database_name="new", collection_name="history"
    ).messages
    return mongo_db_chat_message_history

def get_current_session_history() -> MongoDBChatMessageHistory:
    current_session_id = st.session_state.namespace+"_"+st.session_state.session_id
    return MongoDBChatMessageHistory(
        MONGO_DB_CONN_STR , current_session_id, database_name="new", collection_name="history"
    )