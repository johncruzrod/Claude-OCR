import streamlit as st
import anthropic
from PIL import Image
import io
import base64
from docx import Document
import tempfile

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

def compress_image(image_data, max_size_mb=4.8):
   """Compress image bytes while preserving quality"""
   try:
       # Convert image data to bytes if it isn't already
       if isinstance(image_data, Image.Image):
           img_byte_arr = io.BytesIO()
           image_data.save(img_byte_arr, format='JPEG', quality=95)
           image_data = img_byte_arr.getvalue()
       elif hasattr(image_data, 'read'):
           image_data = image_data.read()
       
       # Calculate target size
       max_bytes = int(max_size_mb * 1024 * 1024)
       
       # If already small enough, just return it
       if len(image_data) <= max_bytes:
           return image_data
       
       # Open image with Pillow
       img = Image.open(io.BytesIO(image_data))
       
       # Convert to RGB if necessary
       if img.mode in ('RGBA', 'LA', 'P'):
           bg = Image.new('RGB', img.size, (255, 255, 255))
           if img.mode == 'RGBA':
               bg.paste(img, mask=img.split()[-1])
           else:
               bg.paste(img)
           img = bg
       
       # Start with original size and high quality
       quality = 95
       width, height = img.size
       
       while True:
           out = io.BytesIO()
           img.save(out, format='JPEG', quality=quality, optimize=True)
           out_size = out.tell()
           
           if out_size <= max_bytes:
               return out.getvalue()
           
           if quality > 30:
               quality -= 5
           else:
               # Scale down the image
               ratio = (max_bytes / out_size) ** 0.5
               width = int(width * ratio)
               height = int(height * ratio)
               img = img.resize((width, height), Image.Resampling.LANCZOS)
               quality = 80  # Reset quality after resize
           
           if width < 800 and height < 800:
               # Final compression attempt
               out = io.BytesIO()
               img.save(out, format='JPEG', quality=30, optimize=True)
               return out.getvalue()
               
   except Exception as e:
       st.error(f"Error compressing image: {str(e)}")
       # Last resort: brutal compression
       try:
           img = Image.open(io.BytesIO(image_data))
           out = io.BytesIO()
           img.convert('RGB').save(out, format='JPEG', quality=20)
           return out.getvalue()
       except:
           raise Exception("Failed to compress image")

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

def process_image(client, image_data):
   """Process image data with Claude Vision API"""
   try:
       # Compress the image
       compressed_data = compress_image(image_data)
       
       # Convert to base64
       img_base64 = base64.b64encode(compressed_data).decode()
       
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
       
       response = client.messages.create(
           model="claude-3-sonnet-20240229",
           max_tokens=4096,
           temperature=0,
           messages=[{
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
                   {"type": "text", "text": prompt}
               ]
           }]
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
           type=['png', 'jpg', 'jpeg', 'heic', 'heif', 'webp', 'bmp', 'tiff'],
           accept_multiple_files=True
       )
       if uploaded_files:
           for file in uploaded_files:
               try:
                   # Store file directly
                   images.append(file)
               except Exception as e:
                   st.error(f"Error loading image: {str(e)}")
   else:
       # Initialize session state for captured images if it doesn't exist
       if 'captured_images' not in st.session_state:
           st.session_state.captured_images = []
           
       # Camera input
       picture = st.camera_input("Take a picture")
       if picture is not None:
           try:
               if 'last_photo' not in st.session_state or picture != st.session_state.last_photo:
                   st.session_state.captured_images.append(picture)
                   st.session_state.last_photo = picture
                   st.rerun()
           except Exception as e:
               st.error(f"Error capturing image: {str(e)}")
               
       images = st.session_state.captured_images.copy()
   
   # Display all images with remove buttons
   if images:
       st.subheader("Preview")
       cols = st.columns(min(len(images), 3))
       for idx, image_data in enumerate(images):
           with cols[idx % 3]:
               # Display image using PIL
               img = Image.open(image_data)
               st.image(img, caption=f"Image {idx + 1}", use_container_width=True)
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
               for idx, image_data in enumerate(images, 1):
                   st.write(f"Processing Image {idx}...")
                   progress_bar = st.progress(0)
                   
                   # Process image
                   text = process_image(client, image_data)
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
