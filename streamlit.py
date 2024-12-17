import streamlit as st
import os
from app import response_generator, ProcessDoucment

# Directory to save uploaded PDFs
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sidebar setup
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/8/89/Palau_de_la_Generalitat_de_Catalunya_1.jpg",
    use_container_width=True,
)

# PDF upload in the sidebar
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

# Save uploaded PDFs and display their names
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

if len(uploaded_files)>0:
            processBtn = st.sidebar.button("Process Selected Files")
            if processBtn:
                # Process the selected files
                with st.spinner("Processing files..."):
                     ProcessDoucment()
# Display the list of uploaded PDFs
st.sidebar.header("Uploaded PDFs")
uploaded_pdf_files = os.listdir(UPLOAD_FOLDER)
if uploaded_pdf_files:
    for pdf_name in uploaded_pdf_files:
        st.sidebar.write(pdf_name)
else:
    st.sidebar.write("No PDFs uploaded yet.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user queries
if prompt := st.chat_input("Ask your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    response = response_generator(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})