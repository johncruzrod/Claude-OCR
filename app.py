import streamlit as st
import anthropic
from PIL import Image
import io
import base64
from docx import Document
import tempfile
import numpy as np
import streamlit.components.v1 as components
import json

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

# Custom HTML/JS for high quality camera capture
CAMERA_HTML = """
<div>
    <video id="video" style="width: 100%; max-width: 800px;" autoplay></video>
    <button id="capture" style="margin: 10px 0;">Capture Photo</button>
    <canvas id="canvas" style="display: none;"></canvas>
</div>

<script>
let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let captureButton = document.getElementById('capture');
let stream = null;

// Request high-resolution camera access
navigator.mediaDevices.getUserMedia({
    video: {
        width: { ideal: 4096 },
        height: { ideal: 2160 },
        facingMode: 'environment'
    }
}).then(function(s) {
    stream = s;
    video.srcObject = s;
}).catch(function(err) {
    console.error("Error accessing camera:", err);
});

captureButton.onclick = function() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    // Convert to high-quality JPEG
    const imageData = canvas.toDataURL('image/jpeg', 1.0);
    
    // Send to Streamlit
    window.parent.postMessage({
        type: 'camera_capture',
        data: imageData
    }, '*');
};

// Cleanup on unmount
window.addEventListener('beforeunload', function() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
</script>
"""

def optimize_image_from_base64(base64_string):
    """Optimize image from base64 while maintaining maximum quality"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate target size maintaining aspect ratio
        max_dimension = 4096  # Support up to 4K resolution
        width, height = img.size
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save with maximum quality first
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95, optimize=True)
        size = output.tell()
        
        # Only reduce quality if absolutely necessary
        if size > 5 * 1024 * 1024:  # 5MB limit
            quality = 90
            while size > 5 * 1024 * 1024 and quality >= 70:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size = output.tell()
                quality -= 5
        
        return base64.b64encode(output.getvalue()).decode()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

def optimize_uploaded_image(file_obj):
    """Optimize uploaded image file"""
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        img = Image.open(file_obj)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Use same logic as base64 optimization
        max_dimension = 4096
        width, height = img.size
        
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=95, optimize=True)
        size = output.tell()
        
        if size > 5 * 1024 * 1024:
            quality = 90
            while size > 5 * 1024 * 1024 and quality >= 70:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size = output.tell()
                quality -= 5
                
        return output.getvalue()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

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

def process_image_with_claude(client, image_data, is_base64=False):
    """Process image with Claude Vision API"""
    try:
        if not is_base64:
            image_base64 = base64.b64encode(image_data).decode()
        else:
            image_base64 = image_data
            
        prompt = """Please accurately transcribe all text from this image.
Output only the transcribed text with no additional commentary.
Preserve formatting and structure where possible.
Pay special attention to handwriting and use context to ensure accuracy."""

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
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
    Upload images or take high-quality photos to extract text using Claude Vision API.
    For best results:
    - Ensure good lighting and focus when taking photos
    - Hold the camera steady and frame the text clearly
    - Avoid glare and shadows on the text
    """)
    
    # Get API key
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    
    # Initialize session state
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
            # Process and store uploaded images
            for file in uploaded_files:
                if file not in [img.get('file') for img in st.session_state.images]:
                    optimized = optimize_uploaded_image(file)
                    st.session_state.images.append({
                        'file': file,
                        'data': optimized,
                        'type': 'uploaded'
                    })
    else:
        st.markdown("""
        ### Camera Tips
        - Use good lighting
        - Hold phone steady
        - Position text parallel to camera
        - Tap to focus on text
        """)
        
        # Custom high-quality camera component
        components.html(CAMERA_HTML, height=600)
        
        # Handle captured photos via streamlit events
        if st.session_state.get('camera_capture'):
            captured_data = st.session_state.camera_capture
            optimized_base64 = optimize_image_from_base64(captured_data)
            
            # Add to images if not already present
            if optimized_base64 not in [img.get('data') for img in st.session_state.images]:
                st.session_state.images.append({
                    'data': optimized_base64,
                    'type': 'captured'
                })
            
            # Clear the capture state
            del st.session_state.camera_capture
    
    # Display and manage images
    if st.session_state.images:
        st.subheader("Images to Process")
        
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.images):
            with cols[idx % 3]:
                if img_data['type'] == 'uploaded':
                    img_data['file'].seek(0)
                    st.image(img_data['file'], caption=f"Image {idx + 1}", use_column_width=True)
                else:
                    st.image(f"data:image/jpeg;base64,{img_data['data']}", 
                            caption=f"Image {idx + 1}", 
                            use_column_width=True)
                    
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
            
            for idx, img_data in enumerate(st.session_state.images, 1):
                progress_text.write(f"Processing Image {idx}/{len(st.session_state.images)}...")
                
                if img_data['type'] == 'uploaded':
                    text = process_image_with_claude(client, img_data['data'])
                else:
                    text = process_image_with_claude(client, img_data['data'], is_base64=True)
                    
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

# Handle camera capture events
if __name__ == "__main__":
    # Initialize session state for camera captures
    if 'camera_capture' not in st.session_state:
        st.session_state.camera_capture = None
        
    # Create a new key in session state to track current camera capture
    current_capture = st.query_params.get("camera_capture")
    if current_capture:
        st.session_state.camera_capture = current_capture
        
    main()
