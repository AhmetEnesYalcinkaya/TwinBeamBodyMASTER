import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import glob
import json

def predict_and_save(uploaded_image):
    model = YOLO("best.pt")
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    uploaded_image.save(temp_image_path)

    # Run prediction
    results = model.predict(source=temp_image_path, save=True, save_txt=True, save_conf=True)

    # Dynamically find the latest prediction folder
    prediction_folders = sorted(glob.glob("runs/detect/predict*"), key=os.path.getmtime, reverse=True)
    if prediction_folders:
        latest_folder = prediction_folders[0]
        pred_image_path = os.path.join(latest_folder, os.path.basename(temp_image_path))
        return pred_image_path, results
    else:
        raise FileNotFoundError("Prediction folder not found.")

# Streamlit UI Setup
st.set_page_config(
    page_title="Spiral Technology",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍",
    menu_items={"Get Help": None, "Report a Bug": None, "About": None}
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Logo Placement
logo_path = "logo.jpg"  # Logo uploaded by user
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=False, width=150)

# Sidebar - Image Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.sidebar.success("Image uploaded successfully!")

    # Layout
    col1, col2 = st.columns(2)
    
    # Original Image
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
    
    # Predict and display the resulting image
    with st.spinner("Running YOLO prediction..."):
        try:
            predicted_image_path, results = predict_and_save(image)
            with col2:
                st.subheader("Predicted Image")
                st.image(predicted_image_path, caption="Predicted Image", use_container_width=True)
                
                # Collect JSON response for all bounding boxes
                detections = []
                for result in results:
                    for box in result.boxes:
                        coords = box.xyxy.tolist()  # [x1, y1, x2, y2]
                        confidence = box.conf.item()
                        cls = box.cls.item()
                        
                        # Calculate width (W) and height (H)
                        x1, y1, x2, y2 = coords[0]
                        width = x2 - x1
                        height = y2 - y1
                        
                        detections.append({
                            "Class": int(cls),
                            "Confidence": round(confidence, 2),
                            "Width": round(width, 0),
                            "Height": round(height, 0)
                        })

                # Convert detections to JSON
                json_data = json.dumps(detections, indent=4)

                # Add a download button for JSON
                st.subheader("Download Detections")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="detections.json",
                    mime="application/json"
                )

        except FileNotFoundError as e:
            st.error(str(e))

    # Clean up temporary image
    if os.path.exists("temp_image.jpg"):
        os.remove("temp_image.jpg")
