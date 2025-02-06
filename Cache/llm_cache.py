from langchain_mongodb.cache import MongoDBCache
import os
import streamlit as st

def get_mongo_cache():
    MONGO_DB_CONN_STR = os.getenv("MONGO_DB_CONN_STR")
    collection_name = st.session_state.index_name+"_"+st.session_state.namespace
    mongo_cache = MongoDBCache( 
        connection_string=MONGO_DB_CONN_STR,
        database_name="new", # new 
        collection_name=collection_name, # userId_dossierName
    )
    return mongo_cache

# CURRENT PROBLEM 
# ---------------

# Gaurav ( Dossier - Gaurav ) ( Annexure-B ) ( hi ) ( Hello, how can I assist you today? ) ( summarize the document ) ( Summary of Annexure J )
# Shikhar ( Dossier - Shikhar ) ( Annexure-A1 ) ( hi ) ( Hello, how can I assist you today? ) ( summarize the document ) ( Summary of Annexure J )
# Ayush ( Dossier - Ayush ) ( Annexure-J ) ( hi ) ( Hello, how can I assist you today? ) ( summarize the document ) ( Summary of Annexure J )

# SOLUTION 1
# ----------

#  collection_name="userId_dossierName",
# Gaurav ( Dossier - Gaurav ) ( Annexure-B ) ( hi ) ( Hello, how can I assist you today? ) ( summarize the document ) ( Summary of Annexure B )
# Shikhar ( Dossier - Shikhar ) ( Annexure-A1 ) ( hi ) ( Hello , how can I help you? ) ( summarize the document ) ( Summary of Annexure A1 )
# Ayush ( Dossier - Ayush ) ( Annexure-J ) ( hi ) ( Namaste, how are you doing? ) ( summarize the document ) ( Summary of Annexure J )

# PROBLEM 2
# ---------

# Gaurav ( Dossier - Gaurav ) ( Annexure-B ) ( hi ) ( Hello, how can I assist you today? ) ( summarize the document ) ( Summary of Annexure B ) [...chat with Annexure B] ( History of India ) ( summarize the document ) ( Summary of Annexure B )
# Shikhar ( Dossier - Shikhar ) ( Annexure-A1 ) ( hi ) ( Hello , how can I help you? ) ( summarize the document ) ( Summary of Annexure A1 ) [...chat with Annexure A1] ( History of America ) ( summarize the document ) ( Summary of Annexure A1 )
# Ayush ( Dossier - Ayush ) ( Annexure-J ) ( hi ) ( Namaste, how are you doing? ) ( summarize the document ) ( Summary of Annexure J ) [...chat with Annexure J] ( History of Africa ) ( summarize the document ) ( Summary of Annexure J )