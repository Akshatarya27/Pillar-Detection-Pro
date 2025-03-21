# Step-by-Step Guide

1. **Launch LabelMe**  
   Open Command Prompt and run:  
   labelme

Import Images
In LabelMe:

Click Open Dir and select your image folder (dataset/images/train or dataset/images/valid).

2. **Label Objects with AI**
For each image:

Click the AI Tool (magic wand icon).

Type pillar (or your object name) in the prompt.

Save annotations as JSON (Ctrl + S).

3. **Convert JSON to YOLO TXT**
Run JSON_to_TXT.ipynb to auto-save labels in:

dataset/labels/train (for training images)

dataset/labels/valid (for validation images)

4. **Train YOLO Model**
Run new_yolo.ipynb to train the model. Trained weights will save to runs/detect/train/weights/best.pt.

5. **Set Up Streamlit App**
Ensure yolo_ui.py points to your trained model:

model = YOLO('runs/detect/train/weights/best.pt')

6. **Run Detection App**
In Command Prompt:

streamlit run yolo_ui.py#