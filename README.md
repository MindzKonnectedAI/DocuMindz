# DocuMindz

DocuMindz is a PDF Retrieval-Augmented Generation (RAG) application developed using Streamlit. It allows users to efficiently search and retrieve information from multiple PDF documents, providing quick answers to their queries based on the content of the documents.


## Features

- **Multiple PDF Support**: Convert multiple PDFs to vectors and search through them simultaneously.

- **Intuitive UI**: A simple and clean interface built using Streamlit, allowing easy interaction with the app.

- **Fast and Accurate**: Leverages Langchain for quick and accurate question answering based on PDF content.

- **Highlighting Feature**: The app highlights the most relevant document in the list, improving the user experience.
  

## Installation

To run DocuMindz locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/DocuMindz.git

2. Navigate to the project directory:
    ```bash
    cd DocuMindz

3. Install the required packages:
   ``` bash
   pip install -r requirements.txt

4. Create a .env file 
   ``` bash
   touch .env

5. Get required api keys ( Refer .env.sample file )

6. Run the Streamlit app 
   ```bash
   streamlit run app.py


## Usage

1. Upload one or more PDFs to the app.

2. Ask questions related to the content of the PDFs.

3. Receive accurate and quick responses with highlighted relevant documents.


## Future Enhancements

1. Advanced Search Features: Implementing advanced filters for more refined searches.
2. Better UI/UX: Continuously improving the user interface for a smoother experience.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.