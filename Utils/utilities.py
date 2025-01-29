import streamlit as st
import time
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from Database.VectorDatabase.pinecone import pc
from Utils.session_states import initialize_session_states

# Initialize session states
initialize_session_states()

# Display a temporary success message
def temporary_success_message(message, duration=2):
    # Show success message
    success = st.success(message)
    # Wait for specified duration
    time.sleep(duration)
    # Clear the success message
    success.empty()

def save_file(save_folder,file):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_path = os.path.join(save_folder, file.name)
    with open(file_path, mode='wb') as w:
        w.write(file.getvalue())
        st.session_state["file_uploader_key"] += 1

# get_session_history function , to be used with RunnableWithMessageHistory class , this is used to pass session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]       

def disable():
    st.session_state.disabled = True
        
def disableOff():
    st.session_state.disabled = False

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

def update_dossier():
    st.session_state.namespace = st.session_state.dossier_radio

# Function to determine the default index
def get_default_index(combined_dossiers):
    return combined_dossiers.index(st.session_state.namespace) if st.session_state.namespace in combined_dossiers else 0
