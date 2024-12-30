import streamlit as st
import streamlit.components.v1 as components
import anthropic
from PIL import Image
import io
import base64
import tempfile
from docx import Document

# Set page config
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

#--------------------------------------------------------------------
# 1. Custom HTML/JS for high-quality camera capture (mobile/desktop)
#--------------------------------------------------------------------
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
        // Attempt to use rear camera with high resolution
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 4096 },
                height: { ideal: 2160 },
                focusMode: { ideal: 'continuous' }
            }
        });
    } catch (err) {
        // Fallback to any camera with lower resolution if necessary
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

    // Handle orientation changes for mobile/desktop
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
        // Match canvas size to the video stream
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current frame from the video
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to a high-quality JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.95);
        
        // Send base64 to parent iframe (Streamlit)
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: imageData
        }, '*');
    };

    // Stop video tracks on exit
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
}

// Start camera on load
initCamera();
</script>
"""

#--------------------------------------------------------------------
# 2. A small function to embed camera HTML and retrieve base64 data
#--------------------------------------------------------------------
def camera_input(key="camera_capture"):
    """
    Renders the custom HTML camera in an iframe
    and returns the base64-encoded JPEG if captured.
    """
    # Hidden HTML snippet that listens for postMessage events and updates
    # a hidden HTML input. This input is tied to a Streamlit text_input with
    # label_visibility='collapsed', so it won't appear on screen.
    st.markdown(
        """
        <script>
        window.addEventListener("message", (event) => {
            if (event.data.type === "streamlit:setComponentValue") {
                const base64Data = event.data.value;
                // Place data in a hidden input with id=hidden_camera_input
                const hiddenInput = document.getElementById("hidden_camera_input");
                if (hiddenInput) {
                    hiddenInput.value = base64Data;
                    hiddenInput.dispatchEvent(new Event('change'));
                }
            }
        });
        </script>
        <input type="hidden" id="hidden_camera_input" name="hidden_camera_input" />
        """,
        unsafe_allow_html=True
    )
    
    # Insert the actual camera UI
    components.html(CAMERA_HTML, height=600)

    # A text input that is never displayed (label_visibility='collapsed') but
    # used to store the captured base64 data. We read from it in Python.
    base64_str = st.text_input("", value="", key=key, label_visibility="collapsed")
    return base64_str

#--------------------------------------------------------------------
# 3. Optimise image for size under 5 MB, preserving quality
#--------------------------------------------------------------------
def optimise_image(input_data, is_base64=False):
    """Optimise image while maintaining quality under ~5MB."""
    try:
        if is_base64:
            # Strip any prefix in the data URL
            if ',' in input_data:
                input_data = input_data.split(',')[1]
            image_data = base64.b64decode(input_data)
            img = Image.open(io.BytesIO(image_data))
        else:
            # File-like object
            if hasattr(input_data, 'seek'):
                input_data.seek(0)
            img = Image.open(input_data)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Scale down if larger than 4096 in either dimension
        max_dim = 4096
        width, height = img.size
        if width > max_dim or height > max_dim:
            if width >= height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save to a buffer at high quality, check size
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95, optimize=True)
        size_bytes = buffer.tell()

        # If still > 5MB, lower quality gradually
        if size_bytes > 5 * 1024 * 1024:
            quality = 90
            while size_bytes > 5 * 1024 * 1024 and quality >= 75:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                size_bytes = buffer.tell()
                quality -= 5

        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

#--------------------------------------------------------------------
# 4. Send image to Claude Vision for transcription
#--------------------------------------------------------------------
def process_image(client, image_data):
    """Send image bytes to Claude Vision API and return transcription."""
    try:
        img_b64 = base64.b64encode(image_data).decode()
        
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
                                "data": img_b64
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

#--------------------------------------------------------------------
# 5. Create a docx with all extracted texts
#--------------------------------------------------------------------
def create_docx(texts):
    """Create a Word document containing all extracted texts."""
    doc = Document()
    doc.add_heading("Extracted Text from Images", 0)
    
    for idx, text in enumerate(texts, start=1):
        doc.add_heading(f"Image {idx}", level=1)
        doc.add_paragraph(text)
        if idx < len(texts):
            doc.add_page_break()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
        doc.save(tmp.name)
        return tmp.name

#--------------------------------------------------------------------
# 6. Main Streamlit application
#--------------------------------------------------------------------
def main():
    st.title("📝 Image Text Extractor")

    # Requires you to have st.secrets["ANTHROPIC_API_KEY"]
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)

    if "images" not in st.session_state:
        st.session_state.images = []

    input_method = st.radio("Choose input method:", ["Upload Image(s)", "Take Photo"])

    # 1) Multiple file uploads
    if input_method == "Upload Image(s)":
        st.info("Upload one or more images for text extraction")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                # Only process if not already in session
                if f not in [img['file'] for img in st.session_state.images if 'file' in img]:
                    try:
                        optimised_data = optimise_image(f, is_base64=False)
                        st.session_state.images.append({
                            "file": f,
                            "data": optimised_data
                        })
                    except Exception as e:
                        st.error(f"Error processing uploaded image: {str(e)}")

    # 2) Take photo via custom HTML camera
    else:
        st.info("Position text clearly in the frame, ensure good lighting, and click 'Take Photo'")
        base64_image = camera_input()

        # If we got new data (starts with 'data:image/jpeg;base64')
        if base64_image and base64_image.startswith("data:image/jpeg;base64"):
            try:
                optimised_data = optimise_image(base64_image, is_base64=True)
                # Avoid duplicates if the same image is captured again
                if optimised_data not in [img.get("data") for img in st.session_state.images]:
                    st.session_state.images.append({"data": optimised_data})
                    # Force a rerun so the image gallery updates immediately
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing captured image: {str(e)}")

    #----------------------------------------------------------------
    # Display images and let user remove or clear them
    #----------------------------------------------------------------
    if st.session_state.images:
        st.subheader("Images to Process")
        columns = st.columns(3)
        for i, img_dict in enumerate(st.session_state.images):
            with columns[i % 3]:
                try:
                    img = Image.open(io.BytesIO(img_dict["data"]))
                    st.image(img, caption=f"Image {i + 1}", use_column_width=True)
                    
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.images.pop(i)
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error displaying image {i + 1}: {str(e)}")

        if len(st.session_state.images) > 1:
            if st.button("Clear All"):
                st.session_state.images = []
                st.experimental_rerun()

        #----------------------------------------------------------------
        # Extract text from images via Claude Vision
        #----------------------------------------------------------------
        if st.button("Extract Text", type="primary"):
            st.subheader("Extracted Text")

            extracted_texts = []
            progress = st.progress(0)
            status = st.empty()

            for idx, img_dict in enumerate(st.session_state.images):
                status.write(f"Processing image {idx + 1} of {len(st.session_state.images)}...")
                text_result = process_image(client, img_dict["data"])
                extracted_texts.append(text_result)

                with st.expander(f"Text from Image {idx + 1}", expanded=True):
                    st.write(text_result)

                progress.progress((idx + 1) / len(st.session_state.images))

            status.write("Processing complete!")

            # Offer a Word document download of all results
            if extracted_texts:
                docx_path = create_docx(extracted_texts)
                with open(docx_path, "rb") as doc_file:
                    st.download_button(
                        label="Download as Word Document",
                        data=doc_file,
                        file_name="extracted_text.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()
