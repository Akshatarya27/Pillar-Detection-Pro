
import streamlit as st  # type: ignore
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from PIL import Image
import pandas as pd

# Load trained model
model = YOLO("runs/detect/train3/weights/best.pt")  # Ensure this path is correct

# Streamlit UI
st.title("YOLOv8 Object Detection App")
st.write("Upload an image to detect pillars using YOLOv8")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Run inference
    results = model(image_np)
    
    # Initialize a list to store pillar coordinates
    pillar_coordinates = []
    processed_image = image_np.copy()  # Copy original image for processing

    # Extract bounding box coordinates for each detected pillar
    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy  # Get coordinates as (x1, y1, x2, y2)
        labels = result.names  # Object class labels
        
        # Loop through each detected object
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box  # Bounding box coordinates
            label = labels[int(result.boxes.cls[i])]  # Object label
            
            # If the label is 'pillar', store the coordinates and draw bounding boxes
            if label == 'pillar':  # Ensure you're detecting 'pillar'
                pillar_id = len(pillar_coordinates) + 1  # Assign unique ID
                
                # Store coordinates
                pillar_coordinates.append({
                    'Pillar': f'Pillar {pillar_id}',
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                })
                
                # Draw bounding box on the processed image (using OpenCV)
                cv2.rectangle(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 🆕 Add a text label to the image
                # cv2.putText(
                #     processed_image,
                #     f"Pillar {pillar_id}",
                #     (int(x1), int(y1)+10),  # Position above the bounding box
                #     cv2.FONT_HERSHEY_DUPLEX,
                #     0.35,  # Font scale
                #     (255, 0, 0),  # Green text
                #     0  # Thickness
                # )

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_container_width=True)

    # Display the processed image with bounding boxes and labels
    st.subheader("Processed Image with Bounding Boxes")
    st.image(processed_image, caption="Processed Image", use_container_width=True)

    # Show pillar coordinates below the processed image
    if pillar_coordinates:
        st.subheader("Pillar Coordinates")
        df = pd.DataFrame(pillar_coordinates)
        st.table(df)  # Display pillar coordinates as a table
        
        # Option to download the processed image with bounding boxes
        output_path = "processed_image.jpg"
        cv2.imwrite(output_path, processed_image)
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Image",
                data=file,
                file_name=output_path,
                mime="image/jpeg"
            )

        # Option to download the coordinates as Excel
        excel_file = "pillar_coordinates.xlsx"
        df.to_excel(excel_file, index=False)
        with open(excel_file, "rb") as file:
            st.download_button(
                label="Download Coordinates as Excel",
                data=file,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.write("No pillars detected in the image.")