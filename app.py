import streamlit as st
import streamlit.components.v1 as components
import anthropic
import base64
import io
import tempfile
from docx import Document
from PIL import Image

# Configure Streamlit page
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

# -------------------------------------------------------------------
# 1. Custom HTML/JS for high-quality camera capture.
# -------------------------------------------------------------------
# Note the large, bright "Capture Photo" button for clarity.
CAMERA_HTML = """
<style>
  /* Centre things more obviously */
  #camera-container {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  #video {
    width: 100%;
    max-width: 600px;
    border: 2px solid #ccc;
    margin-bottom: 1em;
  }
  #capture-btn {
    padding: 1em 2em;
    font-size: 18px;
    background-color: #e02424;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
  }
  #capture-btn:hover {
    background-color: #c81f1f;
  }
  #canvas {
    display: none;
  }
</style>

<div id="camera-container">
    <video id="video" playsinline autoplay></video>
    <button id="capture-btn">Capture Photo</button>
    <canvas id="canvas"></canvas>
</div>

<script>
async function initCamera() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture-btn');
    let stream = null;

    try {
        // Attempt to use rear camera at high resolution (4K)
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 4096 },
                height: { ideal: 2160 },
                focusMode: { ideal: 'continuous' }
            }
        });
    } catch (err) {
        // Fallback to any camera with typical HD resolution
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

    // Assign the stream to the video element
    video.srcObject = stream;
    video.setAttribute("playsinline", true); // iOS compatibility

    // When the user clicks "Capture Photo", draw current frame to canvas
    captureButton.onclick = function() {
        // Match canvas dimensions to the video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas to a JPEG at ~95% quality
        const imageData = canvas.toDataURL('image/jpeg', 0.95);

        // Post the base64 data back to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: imageData
        }, '*');
    };

    // Cleanup on exit
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
}

// Start camera
initCamera();
</script>
"""

# -------------------------------------------------------------------
# 2. Minimal helper that injects the HTML and returns captured base64
# -------------------------------------------------------------------
def take_photo():
    """
    Renders high-res camera feed with a visible "Capture Photo" button.
    Returns the base64 data URL (e.g. 'data:image/jpeg;base64,...') if captured, else None.
    """
    # We place an invisible text_input to store the image data
    # The JS above will send a postMessage to the parent, updating this text_input
    components.html(CAMERA_HTML, height=650, scrolling=False)

    # Show a text input that is linked to the postMessage event from the HTML
    # but hide its label so it doesn't clutter the UI.
    return st.text_input("", value="", key="captured_photo", label_visibility="collapsed")

# -------------------------------------------------------------------
# 3. Optimise image to keep it under ~5MB while preserving quality.
# -------------------------------------------------------------------
def optimise_image(image_data, is_base64=False):
    """
    Takes either base64 or file-like data, ensures it is a JPEG under ~5MB.
    """
    try:
        if is_base64:
            # Strip data URL prefix
            if "," in image_data:
                image_data = image_data.split(",")[1]
            raw_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            if hasattr(image_data, "seek"):
                image_data.seek(0)
            img = Image.open(image_data)

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if above 4096 in either dimension
        max_dim = 4096
        w, h = img.size
        if w > max_dim or h > max_dim:
            if w >= h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Save to buffer at quality=95
        buf = io.BytesIO()
        img.save(buf, format="JPEG", optimize=True, quality=95)
        size_bytes = buf.tell()

        # If >5MB, keep lowering quality
        if size_bytes > 5 * 1024 * 1024:
            quality = 90
            while size_bytes > 5 * 1024 * 1024 and quality >= 75:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", optimize=True, quality=quality)
                size_bytes = buf.tell()
                quality -= 5

        return buf.getvalue()

    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

# -------------------------------------------------------------------
# 4. Send an optimised image to Claude for OCR
# -------------------------------------------------------------------
def process_image_with_claude(client, image_bytes):
    """
    Sends the image to Claude Vision, returns the transcribed text.
    """
    try:
        b64_img = base64.b64encode(image_bytes).decode()
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
                                "data": b64_img
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

# -------------------------------------------------------------------
# 5. Create a docx combining all extracted texts
# -------------------------------------------------------------------
def create_docx(text_list):
    """
    Returns the file path of a docx containing each text under a heading.
    """
    doc = Document()
    doc.add_heading("Extracted Text from Images", 0)

    for i, text in enumerate(text_list, start=1):
        doc.add_heading(f"Image {i}", level=1)
        doc.add_paragraph(text)
        if i < len(text_list):
            doc.add_page_break()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        doc.save(tmp.name)
        return tmp.name

# -------------------------------------------------------------------
# 6. Main Streamlit app
# -------------------------------------------------------------------
def main():
    st.title("📝 Image Text Extractor")

    # Must define st.secrets["ANTHROPIC_API_KEY"]
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error("Please define your Anthropic API key in st.secrets['ANTHROPIC_API_KEY'].")
        return

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    if "images" not in st.session_state:
        st.session_state["images"] = []  # Each item is {"data": <bytes>, "source": <"upload" or "camera">}

    # Choose method
    method = st.radio("Choose input method:", ["Upload Image(s)", "Take Photo"])

    # ----------------------------------------------------------------
    # 1) Upload multiple images
    # ----------------------------------------------------------------
    if method == "Upload Image(s)":
        st.info("Upload one or more images for text extraction.")
        uploaded_files = st.file_uploader(
            "Choose image files",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_files:
            for f in uploaded_files:
                # Only process new images (avoid duplicates)
                if f not in [img.get("file") for img in st.session_state["images"] if img.get("file")]:
                    try:
                        optimised = optimise_image(f, is_base64=False)
                        st.session_state["images"].append({"data": optimised, "file": f, "source": "upload"})
                    except Exception as e:
                        st.error(f"Error processing '{f.name}': {str(e)}")

    # ----------------------------------------------------------------
    # 2) Take photo with custom camera
    # ----------------------------------------------------------------
    else:
        st.info("Position the text clearly, ensure good lighting, then click 'Capture Photo' below.")
        captured_b64 = take_photo()
        if captured_b64 and captured_b64.startswith("data:image/jpeg;base64"):
            try:
                optimised = optimise_image(captured_b64, is_base64=True)
                # Check if we already have this image
                if optimised not in [img["data"] for img in st.session_state["images"]]:
                    st.session_state["images"].append({"data": optimised, "source": "camera"})
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing captured photo: {str(e)}")

    # ----------------------------------------------------------------
    # Display any accumulated images
    # ----------------------------------------------------------------
    if st.session_state["images"]:
        st.subheader("Images to Process")
        cols = st.columns(3)

        for i, img_info in enumerate(st.session_state["images"]):
            with cols[i % 3]:
                try:
                    pil_img = Image.open(io.BytesIO(img_info["data"]))
                    st.image(pil_img, caption=f"Image {i + 1}", use_column_width=True)

                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state["images"].pop(i)
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error displaying image {i + 1}: {str(e)}")

        # Option to clear all images
        if len(st.session_state["images"]) > 1:
            if st.button("Clear All"):
                st.session_state["images"] = []
                st.experimental_rerun()

        # ----------------------------------------------------------------
        # Extract text from all images via Claude Vision
        # ----------------------------------------------------------------
        if st.button("Extract Text", type="primary"):
            st.subheader("Extracted Text")
            extracted_texts = []
            progress_bar = st.progress(0)
            status_message = st.empty()

            for idx, img_info in enumerate(st.session_state["images"]):
                status_message.write(f"Processing image {idx + 1} of {len(st.session_state['images'])}...")
                text_result = process_image_with_claude(client, img_info["data"])
                extracted_texts.append(text_result)

                with st.expander(f"Text from Image {idx + 1}", expanded=True):
                    st.write(text_result)

                progress_bar.progress((idx + 1) / len(st.session_state["images"]))

            status_message.write("Processing complete!")

            # Offer download of all text in one docx
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
