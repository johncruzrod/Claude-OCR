import streamlit as st
import anthropic
from PIL import Image
import io
import base64
from docx import Document
import tempfile
import streamlit.components.v1 as components

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

# Custom HTML/JS for high quality camera capture that works on mobile and desktop
CAMERA_HTML = """
<div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
    <video id="video" playsinline autoplay style="width: 100%; max-width: 100vh; transform: scaleX(1);"></video>
    <canvas id="canvas" style="display: none;"></canvas>
    <button id="capture" style="margin: 10px 0; padding: 10px 20px; font-size: 16px; background-color: #ff4b4b; color: white; border: none; border-radius: 5px; cursor: pointer;">
        Take Photo
    </button>
</div>

<script>
async function initCamera() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    let stream = null;

    try {
        // First try to get the best possible rear camera
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 4096 },
                height: { ideal: 2160 },
                focusMode: { ideal: 'continuous' }
            }
        });
    } catch (err) {
        // Fallback to any available camera
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            });
        } catch (fallbackErr) {
            console.error("Camera access failed:", fallbackErr);
            return;
        }
    }

    video.srcObject = stream;
    video.setAttribute("playsinline", true); // Required for iOS
    
    // Handle orientation changes
    function updateVideoOrientation() {
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        const isPortrait = windowHeight > windowWidth;
        
        if (isPortrait) {
            video.style.width = "100%";
            video.style.maxWidth = "100vw";
        } else {
            video.style.width = "100%";
            video.style.maxWidth = "100vh";
        }
    }

    window.addEventListener('resize', updateVideoOrientation);
    updateVideoOrientation();

    captureButton.onclick = function() {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw the video frame to canvas
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to high-quality JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.95);
        
        // Send to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: imageData
        }, '*');
    };

    // Cleanup
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
}

// Start camera when component loads
initCamera();
</script>
"""

def optimize_image(input_data, is_base64=False):
    """Optimize image while maintaining quality"""
    try:
        if is_base64:
            # Remove data URL prefix if present
            if ',' in input_data:
                input_data = input_data.split(',')[1]
            image_data = base64.b64decode(input_data)
            img = Image.open(io.BytesIO(image_data))
        else:
            if hasattr(input_data, 'seek'):
                input_data.seek(0)
            img = Image.open(input_data)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate optimal size while maintaining aspect ratio
        max_dimension = 4096  # Support up to 4K
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
        
        # Only reduce quality if absolutely necessary
        if size > 5 * 1024 * 1024:  # 5MB limit
            quality = 90
            while size > 5 * 1024 * 1024 and quality >= 75:
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                size = output.tell()
                quality -= 5
        
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

def process_image(client, image_data):
    """Process image with Claude Vision API"""
    try:
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode()
        
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

def main():
    st.title("📝 Image Text Extractor")
    
    # Get API key
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    
    # Initialize session state
    if 'images' not in st.session_state:
        st.session_state.images = []
    
    # Image input options
    input_method = st.radio("Choose input method:", ["Upload Image(s)", "Take Photo"])
    
    if input_method == "Upload Image(s)":
        st.info("Upload one or more images to extract text")
        uploaded_files = st.file_uploader(
            "Choose image files", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                # Only process new images
                if file not in [img['file'] for img in st.session_state.images if 'file' in img]:
                    try:
                        optimized = optimize_image(file)
                        st.session_state.images.append({
                            'file': file,
                            'data': optimized
                        })
                    except Exception as e:
                        st.error(f"Error processing uploaded image: {str(e)}")
    else:
        st.info("Position text clearly in frame and ensure good lighting")
        
        # Custom camera component
        camera_result = components.html(CAMERA_HTML, height=600)
        
        # Handle camera capture
        if camera_result:
            try:
                optimized = optimize_image(camera_result, is_base64=True)
                if optimized not in [img.get('data') for img in st.session_state.images]:
                    st.session_state.images.append({
                        'data': optimized
                    })
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing captured image: {str(e)}")
    
    # Display and manage images
    if st.session_state.images:
        st.subheader("Images to Process")
        
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.images):
            with cols[idx % 3]:
                try:
                    # Convert image data to format suitable for st.image
                    img = Image.open(io.BytesIO(img_data['data']))
                    st.image(img, caption=f"Image {idx + 1}", use_column_width=True)
                    
                    if st.button(f"Remove", key=f"remove_{idx}"):
                        st.session_state.images.pop(idx)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error displaying image {idx + 1}: {str(e)}")
        
        if len(st.session_state.images) > 1:
            if st.button("Clear All"):
                st.session_state.images = []
                st.rerun()
        
        # Process images
        if st.button("Extract Text", type="primary"):
            st.subheader("Extracted Text")
            
            extracted_texts = []
            progress = st.progress(0)
            status = st.empty()
            
            for idx, img_data in enumerate(st.session_state.images):
                status.write(f"Processing image {idx + 1} of {len(st.session_state.images)}...")
                
                text = process_image(client, img_data['data'])
                extracted_texts.append(text)
                
                with st.expander(f"Text from Image {idx + 1}", expanded=True):
                    st.write(text)
                
                progress.progress((idx + 1) / len(st.session_state.images))
            
            status.write("Processing complete!")
            
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
