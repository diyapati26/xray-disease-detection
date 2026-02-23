import streamlit as st
import torch
import torchvision.transforms as transforms
from torch import nn
import pickle
import numpy as np
from PIL import Image
from torchvision.models import vit_b_16


LABEL_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]


@st.cache_resource
def load_vit():
    vit = vit_b_16(pretrained=True)
    vit.heads = nn.Identity()  
    vit.eval()
    return vit

def load_final_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


def main():
    st.title("Chest X-ray Disease Prediction App")

    vit = load_vit()
    final_model = load_final_model("model3.pickle")

    uploaded_file = st.file_uploader("Upload an X-ray Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    with st.form("clinical_form"):
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=50)
        gender = st.radio("Patient Gender", ["Male", "Female"])
        pa_position = st.checkbox("PA View Position", value=True)
        pixel_spacing_input = st.text_input("Pixel Spacing (e.g., 0.5, 0.5)", "1.0, 1.0")
        img_width = st.number_input("Image Width", min_value=1, value=224)
        img_height = st.number_input("Image Height", min_value=1, value=224)
        submitted = st.form_submit_button("Predict")

    if submitted and uploaded_file:
        with st.spinner("Processing..."):
            try:
                image = Image.open(uploaded_file).convert("RGB")

                pixel_spacing = [float(x) for x in pixel_spacing_input.split(",")]

                st.image(image, caption="Uploaded Image", use_column_width=True)

                with torch.no_grad():
                    img_tensor = preprocess_image(image)
                    img_features = vit(img_tensor).flatten().numpy()

                clinical_features = [
                    age,
                    img_width,
                    img_height,
                    pixel_spacing[0],
                    pixel_spacing[1],
                    1 if pa_position else 0,
                    1 if gender == "Male" else 0
                ]

                combined = np.concatenate([ clinical_features, img_features])
                with torch.no_grad():

                    y_pred = final_model.predict(combined.reshape(1, -1))[0]

                y_pred_bin = (y_pred > 0.5).astype(int)

                predicted_diseases = [LABEL_CLASSES[i] for i, val in enumerate(y_pred_bin) if val == 1 and LABEL_CLASSES[i] != "No Finding"]

                st.subheader("Prediction Results")
                st.write("**Probabilities per Disease:**")
                st.json({LABEL_CLASSES[i]: float(f"{prob:.4f}") for i, prob in enumerate(y_pred)})

                st.write("**Detected Diseases (threshold > 0.4):**")
                if predicted_diseases:
                    st.success(", ".join(predicted_diseases))
                elif y_pred_bin[LABEL_CLASSES.index("No Finding")] == 1:
                    st.success("No Finding")
                else:
                    st.info("No disease detected with current threshold.")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    main()
