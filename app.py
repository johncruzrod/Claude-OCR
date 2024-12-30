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

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
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
    
    # Define the British English prompt for OCR
    prompt = """Please analyse this image and extract all visible text content precisely as it appears. 
    Format the output maintaining the original structure and layout where possible.
    If there are multiple sections or paragraphs, please preserve them.
    Include any relevant formatting notes (e.g., 'Header:', 'Footer:', 'Margin note:') where applicable.
    If any text is unclear or partially visible, please indicate this with [unclear] or [partially visible].
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
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
        picture = st.camera_input("Take a picture")
        if picture:
            image = Image.open(picture)
            images.append(image)
    
    if images:
        st.subheader("Uploaded/Captured Images")
        cols = st.columns(min(len(images), 3))
        for idx, image in enumerate(images):
            cols[idx % 3].image(image, caption=f"Image {idx + 1}", use_column_width=True)
        
        if st.button("Extract Text", type="primary"):
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
