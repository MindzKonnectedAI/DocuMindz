import os
import pymupdf4llm
from Utils.session_states import initialize_session_states
# functions for extracting images from pdf
from PDFParser.PyMuPDF.extract_pdf_images import extract_images_and_text,get_unique_xref_images_and_text

# Initialize session states
initialize_session_states()

def parse_pdf(file_path, user_folder):
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
        
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
            
        return [], None
      