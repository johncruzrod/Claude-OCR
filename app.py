import streamlit as st
import anthropic
from PIL import Image
import io
import base64
from docx import Document
import tempfile
import numpy as np

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

def optimize_image(file_obj):
    """Optimize image for Claude Vision API while maintaining quality"""
    try:
        # Reset file pointer and open image
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        
        img = Image.open(file_obj)
        img = img.convert('RGB')
        
        # Calculate target size while maintaining aspect ratio
        max_dimension = 2048  # Max dimension for good quality
        width, height = img.size
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save with optimal quality
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95, optimize=True)
        size = output.tell()
        
        # If still over 5MB, gradually reduce quality while keeping size
        if size > 5 * 1024 * 1024:
            quality = 90
            while size > 5 * 1024 * 1024 and quality >= 60:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size = output.tell()
                quality -= 5
        
        return output.getvalue()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

def image_to_base64(file_obj):
    """Convert image to base64 string"""
    optimized = optimize_image(file_obj)
    return base64.b64encode(optimized).decode()

def create_docx(texts):
    """Create a Word document from the extracted texts"""
    doc = Document()
    doc.add_heading('Extracted Text from Images', 0)
    
    for idx, text in enumerate(texts, 1):
        doc.add_heading(f'Image {idx}', level=1)
        doc.add_paragraph(text)
        if idx < len(texts):
            doc.add_page_break()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        doc.save(tmp.name)
        return tmp.name

def process_image(client, image):
    """Process a single image with Claude Vision API"""
    img_base64 = image_to_base64(image)
    
    prompt = """Please accurately transcribe all text from this image.
Output only the transcribed text with no additional commentary.
Preserve formatting and structure where possible.
Pay special attention to handwriting and use context to ensure accuracy."""
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0.0,  # Lower temperature for more accurate transcription
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
    Upload images or take photos to extract text using Claude Vision API.
    For best results:
    - Ensure good lighting and focus when taking photos
    - Hold the camera steady and frame the text clearly
    - Avoid glare and shadows on the text
    """)
    
    # Get API key
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    
    # Initialize session state for images
    if 'images' not in st.session_state:
        st.session_state.images = []
    
    # Image input options
    input_method = st.radio("Choose input method:", ["Upload Image(s)", "Take Photo"])
    
    if input_method == "Upload Image(s)":
        uploaded_files = st.file_uploader(
            "Choose image files", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.images = uploaded_files
    else:
        # Camera settings for better quality
        st.markdown("""
        ### Camera Tips
        - Hold your device steady
        - Ensure text is well-lit and in focus
        - Position text parallel to camera
        """)
        
        photo = st.camera_input(
            "Take a photo",
            help="Position the text clearly in frame and ensure good lighting"
        )
        
        if photo and photo not in st.session_state.images:
            st.session_state.images.append(photo)
    
    # Display and manage images
    if st.session_state.images:
        st.subheader("Images to Process")
        
        cols = st.columns(3)
        for idx, img in enumerate(st.session_state.images):
            with cols[idx % 3]:
                img.seek(0)
                st.image(img, caption=f"Image {idx + 1}", use_column_width=True)
                if st.button(f"Remove Image {idx + 1}", key=f"remove_{idx}"):
                    st.session_state.images.pop(idx)
                    st.rerun()
        
        if len(st.session_state.images) > 1:
            if st.button("Clear All Images"):
                st.session_state.images = []
                st.rerun()
        
        # Process images
        if st.button("Extract Text", type="primary"):
            st.subheader("Extracted Text")
            
            extracted_texts = []
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, img in enumerate(st.session_state.images, 1):
                progress_text.write(f"Processing Image {idx}/{len(st.session_state.images)}...")
                
                text = process_image(client, img)
                extracted_texts.append(text)
                
                with st.expander(f"Text from Image {idx}", expanded=True):
                    st.write(text)
                
                progress_bar.progress(idx / len(st.session_state.images))
            
            progress_text.write("Processing complete!")
            
            # Create download option
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
