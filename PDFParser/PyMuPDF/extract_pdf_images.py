import fitz
from PIL import Image
import io
import base64
from langchain.schema.document import Document
from Utils.session_states import initialize_session_states

# Initialize session states
initialize_session_states()

def extract_images_and_text(file_path):
    doc = fitz.open(file_path)
    images_and_text = []

    for page in doc:
        text = page.get_text()
        # print("text :",text)
        image_list = page.get_images(full=True)
        # print("image_list :",image_list)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # words = text.split()
            # text_before_image = " ".join(words[-200:]) if len(words) > 200 else text

            # page text becomes the image context
            text_before_image = text

            images_and_text.append({
              "xref":xref,
              "base64_image": img_base64,
              "text": text_before_image
        })
    
    return images_and_text 
 
# Create a new array with unique xrefs
def get_unique_xref_images_and_text(pdf_image_and_text_array):
    unique_xref_dict = {}
    unique_xref_array = []

    for entry in pdf_image_and_text_array:
        xref = entry["xref"]
        if xref not in unique_xref_dict:
            unique_xref_dict[xref] = entry
            unique_xref_array.append(entry) #as it is

    return unique_xref_array

def convert_image_array_to_documents(unique_xref_array):
    """
    Converts an array of image objects to LangChain Document instances.
    
    Args:
        image_array: List of dictionaries containing 'xref', 'base64_image', and 'text'.

    Returns:
        List of LangChain Document instances.
    """
    documents = []
    for item in unique_xref_array:
        try:
            # Extract fields from the item
            page_content = item.get("base64_image", "")
            metadata = {
                "xref": item.get("xref", ""),
                "surrounding_text": item.get("text", "")
            }
            # Create a Document instance
            document = Document(page_content=page_content, metadata=metadata)
            documents.append(document)
        except Exception as e:
            print(f"Error converting item to Document: {e}")
    return documents