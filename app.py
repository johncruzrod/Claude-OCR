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

def compress_image(image, max_size_mb=4.8):
    """Compress image while maintaining quality and keeping size under max_size_mb"""
    img_byte_arr = io.BytesIO()
    
    # Convert RGBA to RGB if necessary
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    
    # First try: compression with quality reduction
    quality = 85
    width, height = image.size
    
    while True:
        img_byte_arr = io.BytesIO()
        # If image is large, resize first
        if width * height > 1500000:  # If more than ~1.5 megapixels
            scale = min(1, (1500000 / (width * height)) ** 0.5)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            resized_image = image
            
        resized_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        size_mb = len(img_byte_arr.getvalue()) / (1024 * 1024)
        
        if size_mb < max_size_mb:
            break
            
        if quality > 20:
            quality -= 10
        else:
            # If still too big, reduce size
            width = int(width * 0.8)
            height = int(height * 0.8)
            
        if width < 200 or height < 200:  # Prevent images becoming too small
            # Final attempt with minimum size and quality
            img_byte_arr = io.BytesIO()
            resized_image = image.resize((200, int(200 * height/width)), Image.Resampling.LANCZOS)
            resized_image.save(img_byte_arr, format='JPEG', quality=20, optimize=True)
            break
    
    img_byte_arr.seek(0)
    return Image.open(img_byte_arr)

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    # Compress image before converting to base64
    compressed_image = compress_image(image)
    
    # Keep trying compression until we're under limit
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        buffered = io.BytesIO()
        compressed_image.save(buffered, format='JPEG', quality=85-attempt*20)
        size_mb = len(buffered.getvalue()) / (1024 * 1024)
        
        if size_mb < 4.8:  # Safely under 5MB limit
            break
        attempt += 1
        
    buffered.seek(0)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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
   
   # Define the prompt for OCR
   prompt = """You are an expert OCR system. Your task is to accurately transcribe text from this image:

1. Output the exact text you see, preserving spelling and capitalization
2. Preserve line breaks and spacing as they appear
3. Ignore formatting descriptors - just give the raw text
4. Use [unclear] only when text is truly unreadable
5. Include any numbers or special characters exactly as shown
6. Do not add descriptions, headers, or explanations
7. Skip any image descriptions or visual elements
8. Do not include any metadata or context notes

Begin transcription:"""
   
   try:
       response = client.messages.create(
           model="claude-3-sonnet-20240229",
           max_tokens=4096,
           temperature=0,
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
