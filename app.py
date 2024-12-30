import streamlit as st
import anthropic
from PIL import Image
import io
import base64
from docx import Document
import tempfile
import cv2
import numpy as np

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

def compress_image(image_data, max_size_mb=4.8):
   """Compress any image type while preserving quality"""
   try:
       # Convert image data to bytes if it isn't already
       if isinstance(image_data, Image.Image):
           img = image_data
       else:
           # Reset file pointer if it's a file-like object
           if hasattr(image_data, 'seek'):
               image_data.seek(0)
           img = Image.open(image_data)
       
       # Convert to RGB if necessary
       if img.mode in ('RGBA', 'LA', 'P'):
           bg = Image.new('RGB', img.size, (255, 255, 255))
           if img.mode == 'RGBA':
               bg.paste(img, mask=img.split()[-1])
           else:
               bg.paste(img)
           img = bg

       # Start compression
       buffer = io.BytesIO()
       img.save(buffer, format='JPEG', quality=95, optimize=True)
       size = buffer.tell()
       max_bytes = int(max_size_mb * 1024 * 1024)

       # If size is already ok, return
       if size <= max_bytes:
           buffer.seek(0)
           return buffer.getvalue()

       # Gradually reduce quality and resize if needed
       quality = 95
       width, height = img.size
       
       while True:
           buffer = io.BytesIO()
           img.save(buffer, format='JPEG', quality=quality, optimize=True)
           size = buffer.tell()
           
           if size <= max_bytes:
               buffer.seek(0)
               return buffer.getvalue()
           
           if quality > 30:
               quality -= 5
           else:
               # Scale down image
               scale = (max_bytes / size) ** 0.5 * 0.9  # 0.9 for safety margin
               width = int(width * scale)
               height = int(height * scale)
               img = img.resize((width, height), Image.Resampling.LANCZOS)
               quality = 80  # Reset quality after resize

   except Exception as e:
       st.error(f"Error compressing image: {str(e)}")
       raise

def image_to_base64(image):
   """Convert any image to base64 string."""
   try:
       compressed_data = compress_image(image)
       return base64.b64encode(compressed_data).decode()
   except Exception as e:
       st.error(f"Error converting image to base64: {str(e)}")
       raise

def create_docx(texts):
   """Create a Word document from the extracted texts."""
   doc = Document()
   doc.add_heading('Extracted Text from Images', 0)
   
   for idx, text in enumerate(texts, 1):
       doc.add_heading(f'Image {idx}', level=1)
       doc.add_paragraph(text)
       if idx < len(texts):  # Don't add a page break after the last text
           doc.add_page_break()
   
   # Save to temporary file
   with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
       doc.save(tmp.name)
       return tmp.name

def process_image(client, image):
   """Process a single image with Claude Vision API."""
   # Convert to base64
   img_base64 = image_to_base64(image)
   
   # Define the British English prompt for OCR
   prompt = """You are an expert image-to-text AI assistant. Your task is to accurately transcribe text from images.

Take the following image, and output all of the text contexts in plain text, without any additional output (introduction messages etc.)

Aim for high accuracy, especially with things like handwriting, using contextual awareness from the entire image to transcribe word for word the contents."""
   
   try:
       response = client.messages.create(
           model="claude-3-sonnet-20240229",
           max_tokens=4096,
           temperature=0.8,
           messages=[
               {
                   "role": "user",
                   "content": [
                       {
                           "type": "image",
                           "source": {
                               "type": "base64",
                               "media_type": "image/jpeg",
                               "data": img_base64
                           }
                       },
                       {
                           "type": "text",
                           "text": prompt
                       }
                   ]
               }
           ]
       )
       return response.content[0].text
   except Exception as e:
       return f"Error processing image: {str(e)}"

def main():
   st.title("📝 Image Text Extractor")
   st.write("""
   Upload images or take photos to extract text content using Claude's Vision API.
   The extracted text will maintain its original structure and can be downloaded as a Word document.
   """)
   
   # Get API key from secrets
   api_key = st.secrets["ANTHROPIC_API_KEY"]
   
   # Initialize Claude client
   client = anthropic.Anthropic(api_key=api_key)
   
   # Image upload/capture section
   st.subheader("Upload or Capture Images")
   upload_option = st.radio(
       "Choose input method:",
       ["Upload Image(s)", "Take Photo(s)"]
   )
   
   images = []
   if upload_option == "Upload Image(s)":
       uploaded_files = st.file_uploader(
           "Choose image file(s)", 
           type=['png', 'jpg', 'jpeg'], 
           accept_multiple_files=True
       )
       if uploaded_files:
           for file in uploaded_files:
               image = Image.open(file)
               images.append(image)
   else:
       # Initialize session state for captured images if it doesn't exist
       if 'captured_images' not in st.session_state:
           st.session_state.captured_images = []
           
       # Camera input
       picture = st.camera_input("Take a picture")
       if picture is not None:
           try:
               image = Image.open(picture)
               if 'last_photo' not in st.session_state or picture != st.session_state.last_photo:
                   st.session_state.captured_images.append(image)
                   st.session_state.last_photo = picture
                   st.rerun()
           except Exception as e:
               st.error(f"Error capturing image: {str(e)}")
               
       images = st.session_state.captured_images.copy()
   
   # Display all images with remove buttons
   if images:
       st.subheader("Preview")
       cols = st.columns(min(len(images), 3))
       for idx, image in enumerate(images):
           with cols[idx % 3]:
               st.image(image, caption=f"Image {idx + 1}", use_container_width=True)
               if upload_option == "Take Photo(s)":
                   if st.button(f"❌ Remove", key=f"remove_{idx}"):
                       st.session_state.captured_images.pop(idx)
                       st.rerun()
       
       # Show clear all button only for camera mode
       if upload_option == "Take Photo(s)" and len(images) > 1:
           if st.button("Clear All Photos", type="secondary"):
               st.session_state.captured_images = []
               st.session_state.last_photo = None
               st.rerun()
       
       # Extract text button
       if st.button("Extract Text", type="primary", key="extract"):
           st.subheader("Extracted Text")
           
           # Process each image
           with st.spinner("Processing images..."):
               extracted_texts = []
               for idx, image in enumerate(images, 1):
                   st.write(f"Processing Image {idx}...")
                   progress_bar = st.progress(0)
                   
                   # Process image
                   text = process_image(client, image)
                   extracted_texts.append(text)
                   
                   # Display extracted text
                   with st.expander(f"Text from Image {idx}", expanded=True):
                       st.write(text)
                   
                   progress_bar.progress(idx / len(images))
               
               # Create download button for Word document
               if extracted_texts:
                   docx_path = create_docx(extracted_texts)
                   with open(docx_path, "rb") as file:
                       st.download_button(
                           label="Download as Word Document",
                           data=file,
                           file_name="extracted_text.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                       )

if __name__ == "__main__":
   main()
