from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Models.chat import llm
import os
from openai import Client
from Utils.session_states import initialize_session_states
from dotenv import load_dotenv  # .env file loading
load_dotenv(override=True)

# Initialize session states
initialize_session_states()

def create_text_summaries(docs):
    print("inside create_text_summaries()")
    prompt_text = """
      You are an assistant tasked with summarizing tables and text.
      Give a concise summary of the table or text.

      Respond only with the summary, no additionnal comment.
      Do not start your message by saying "Here is a summary" or anything like that.
      Just give the summary as it is.

      Table or text chunk: {element} 

                 """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    summarize_chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()
    # Extract content from Document objects
    text_chunks = [doc.page_content for doc in docs]

    # Summarize text
    text_summaries = summarize_chain.batch(text_chunks, {"max_concurrency": 3}) #docs
    return text_summaries

def generate_caption_for_image(base_img, query):
    messages = []
    system_role = {
            "role": "system",
            "content": '''You are a highly advanced multimodal assistant specializing in generating concise and accurate textual summaries for images. Your role is to analyze the visual content of images, including objects, scenes, actions, and emotions, and describe them in a way that captures the essence of the image.

            Each summary should:
            1. Be no longer than 2-3 sentences.
            2. Focus on key elements in the image (e.g., objects, settings, interactions, and emotions).
            3. Avoid unnecessary details or speculative information.
            4. Use formal and neutral language suitable for embedding generation.

            Your output will be used to generate embeddings for semantic search and vector storage, so ensure the summaries are informative and contextually rich.'''
    }
    messages.append(system_role)
    messages.append({"role": "user", "content": f"this is the text... {query}"})
    client = Client(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[*messages,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base_img}",
                        "detail": "high"
                       }
                   }
               ]
           }
       ],
       temperature=0.5,
       top_p=0.9,
       presence_penalty=0.6,
       frequency_penalty=0.5
    )
    assistant_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message       

def create_image_summaries(unique_xref_array):
    print('inside create_image_summaries()')
    image_summaries = []
    for item in unique_xref_array:
       caption = generate_caption_for_image(item['base64_image'], item['text'])
       image_summaries.append(caption)
    return image_summaries

