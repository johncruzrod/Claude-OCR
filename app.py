import streamlit as st
import streamlit.components.v1 as components
import anthropic
import base64
import io
import tempfile
from docx import Document
from PIL import Image

# Configure Streamlit
st.set_page_config(page_title="Image Text Extractor", page_icon="📝", layout="wide")

# --------------------------------------------------------------------
# 1. High-resolution camera HTML with no button inside
#    We rely on a postMessage to trigger the capture
# --------------------------------------------------------------------
CAMERA_HTML = """
<style>
  #camera-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  video {
    width: 100%;
    max-width: 600px;
    border: 2px solid #ccc;
    margin-bottom: 1em;
  }
  canvas {
    display: none;
  }
</style>
<div id="camera-wrap">
  <video id="video" playsinline autoplay></video>
  <canvas id="canvas"></canvas>
</div>
<script>
let stream = null;

async function initCamera() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');

    try {
        // Attempt to get a high-resolution rear camera
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' },
                width: { ideal: 4096 },
                height: { ideal: 2160 },
                focusMode: { ideal: 'continuous' }
            }
        });
    } catch (err) {
        // Fallback to any camera if the above fails
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
    video.setAttribute("playsinline", true);

    // Listen for a message from the parent that says "capturePhoto"
    window.addEventListener('message', (event) => {
        if (event.data === 'capturePhoto') {
            // Capture the current frame
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert canvas to a JPEG at 95% quality
            const imageData = canvas.toDataURL('image/jpeg', 0.95);

            // Send it back to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: imageData
            }, '*');
        }
    });

    // Clean up on page unload
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

# --------------------------------------------------------------------
# 2. Optimise image to be under ~5 MB while preserving quality
# --------------------------------------------------------------------
def optimise_image(image_data, is_base64=False):
    try:
        if is_base64:
            # Strip data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            raw_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(raw_bytes))
        else:
            # It's a file-like
            if hasattr(image_data, "seek"):
                image_data.seek(0)
            img = Image.open(image_data)

        # Convert to RGB if not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Scale down if bigger than 4096 in either dimension
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

        # Save at quality=95
        buf = io.BytesIO()
        img.save(buf, format="JPEG", optimize=True, quality=95)
        size_bytes = buf.tell()

        # If still above 5 MB, lower quality in steps
        if size_bytes > 5 * 1024 * 1024:
            quality = 90
            while size_bytes > 5 * 1024 * 1024 and quality >= 75:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", optimize=True, quality=quality)
                size_bytes = buf.tell()
                quality -= 5

        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        raise Exception(f"Image optimization failed: {str(e)}")

# --------------------------------------------------------------------
# 3. Send image to Claude Vision for transcription
# --------------------------------------------------------------------
def process_image_with_claude(client, image_bytes):
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
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

# --------------------------------------------------------------------
# 4. Create a Word doc containing transcriptions
# --------------------------------------------------------------------
def create_docx(text_list):
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

# --------------------------------------------------------------------
# 5. Main Streamlit app
# --------------------------------------------------------------------
def main():
    st.title("Image Text Extractor")

    # Ensure we have the Anthropic API key
    if "ANTHROPIC_API_KEY" not in st.secrets:
        st.error("Please add your Anthropic API key in st.secrets['ANTHROPIC_API_KEY'].")
        return
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    # Prepare state for images
    if "images" not in st.session_state:
        st.session_state["images"] = []  # each is {"data": bytes, "method": "upload" or "camera"}

    # Let user choose method
    method = st.radio("Choose input method:", ["Upload Image(s)", "Take Photo"])

    # -------------------------
    # 1) File uploader approach
    # -------------------------
    if method == "Upload Image(s)":
        st.info("Upload one or more images for text extraction.")
        uploaded = st.file_uploader(
            "Select images",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            for f in uploaded:
                # Avoid duplicates
                if f not in [x.get("file") for x in st.session_state["images"] if x.get("file")]:
                    try:
                        optimised_data = optimise_image(f, is_base64=False)
                        st.session_state["images"].append({
                            "data": optimised_data,
                            "file": f,
                            "method": "upload"
                        })
                    except Exception as e:
                        st.error(f"Error processing file {f.name}: {str(e)}")

    # -------------------------
    # 2) Camera approach
    # -------------------------
    else:
        st.info("Ensure good lighting and position the camera. Press 'Capture Photo' to take a snapshot.")

        # Show the camera feed with no internal button
        components.html(
            CAMERA_HTML,
            height=550,
            scrolling=False,
            key="my_camera"  # This is important so Streamlit can store the base64
        )

        # This is the external capture button
        if st.button("Capture Photo"):
            # Send a message to the iframe to capture
            # This runs in the parent's DOM context, which holds the iframe
            # We search for the <iframe> whose key is "my_camera"
            js = """
            <script>
            const frames = window.top.document.getElementsByTagName('iframe');
            for (let f of frames) {
                // Attempt to find the one matching the Streamlit key (my_camera).
                // In practice, we check the srcdoc or name. 
                // We'll assume the first one is correct if there's only one. 
                // If multiple iframes exist, you'd need a more robust check.
                if (f.srcdoc && f.srcdoc.indexOf('initCamera') !== -1) {
                    f.contentWindow.postMessage('capturePhoto','*');
                    break;
                }
            }
            </script>
            """
            st.markdown(js, unsafe_allow_html=True)

        # If the iframe returned new data, it will be in st.session_state["my_camera"]
        # because we used key="my_camera"
        captured_data = st.session_state.get("my_camera")
        if captured_data and captured_data.startswith("data:image/jpeg;base64"):
            # Optimise
            try:
                optimised_bytes = optimise_image(captured_data, is_base64=True)
                # Avoid duplicates
                if optimised_bytes not in [x["data"] for x in st.session_state["images"]]:
                    st.session_state["images"].append({
                        "data": optimised_bytes,
                        "method": "camera"
                    })
                    # Clear out st.session_state["my_camera"] so we can capture again
                    st.session_state["my_camera"] = ""
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing captured photo: {str(e)}")

    # -----------------------------------------------
    # Display images and let user remove or clear them
    # -----------------------------------------------
    if st.session_state["images"]:
        st.subheader("Images to Process")

        columns = st.columns(3)
        for i, img_info in enumerate(st.session_state["images"]):
            with columns[i % 3]:
                try:
                    pil_img = Image.open(io.BytesIO(img_info["data"]))
                    st.image(pil_img, caption=f"Image {i + 1}", use_column_width=True)

                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state["images"].pop(i)
                        st.experimental_rerun()

                except Exception as e:
                    st.error(f"Error displaying image {i + 1}: {str(e)}")

        if len(st.session_state["images"]) > 1:
            if st.button("Clear All"):
                st.session_state["images"] = []
                st.experimental_rerun()

        # -------------------------
        # Extract text via Claude
        # -------------------------
        if st.button("Extract Text", type="primary"):
            st.subheader("Extracted Text")

            results = []
            progress_bar = st.progress(0)
            status_box = st.empty()

            for idx, img_dict in enumerate(st.session_state["images"]):
                status_box.write(f"Processing image {idx + 1} of {len(st.session_state['images'])}...")
                text_out = process_image_with_claude(client, img_dict["data"])
                results.append(text_out)

                with st.expander(f"Text from Image {idx + 1}", expanded=True):
                    st.write(text_out)

                progress_bar.progress((idx + 1) / len(st.session_state["images"]))

            status_box.write("Processing complete!")

            # Download all texts as docx
            if results:
                docx_file = create_docx(results)
                with open(docx_file, "rb") as f:
                    st.download_button(
                        label="Download as Word Document",
                        data=f,
                        file_name="extracted_text.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

if __name__ == "__main__":
    main()
