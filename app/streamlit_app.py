"""Interactive Streamlit demo for brain MRI tumor classification."""

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGENET_MEAN, IMAGENET_STD, load_config
from src.gradcam import GradCAM, get_gradcam_target_layer, overlay_heatmap
from src.utils import build_model, get_device, load_checkpoint

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

st.set_page_config(page_title="NeuroScan", page_icon="🧠", layout="wide")

st.title("NeuroScan")
st.caption("Brain MRI tumor classification with Grad-CAM explainability")

config = load_config()
device = get_device()
checkpoint_path = (
    PROJECT_ROOT / config["output_dir"] / "checkpoints" / config["checkpoint_name"]
)
image_size = config.get("image_size", 224)

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


@st.cache_resource
def load_model():
    if not checkpoint_path.exists():
        return None, None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint["config"].get("model", config["model"])
    class_names = checkpoint.get("class_names", config["class_names"])
    model = build_model(model_name, config["num_classes"])
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model, class_names, model_name


model, class_names, model_name = load_model()

if model is None:
    st.warning("No trained model found. Download the dataset and train first.")
    st.code(
        "pip install -r requirements.txt\n"
        "python scripts/prepare_data.py --source <dataset-path>\n"
        "python train.py\n"
        "streamlit run app/streamlit_app.py",
        language="bash",
    )
    st.stop()

tab_classify, tab_about = st.tabs(["Classify MRI", "About"])

with tab_classify:
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with col_upload:
            st.image(image, caption="Input MRI", use_container_width=True)

        if st.button("Analyze", type="primary"):
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]

            top_idx = probabilities.argmax().item()
            predicted = class_names[top_idx]
            display_name = CLASS_LABELS.get(predicted, predicted)

            with col_result:
                st.success(f"**Prediction:** {display_name}")
                st.metric("Confidence", f"{probabilities[top_idx].item() * 100:.1f}%")

                st.subheader("Class probabilities")
                for idx, name in enumerate(class_names):
                    label = CLASS_LABELS.get(name, name)
                    st.progress(
                        float(probabilities[idx]),
                        text=f"{label}: {probabilities[idx].item() * 100:.1f}%",
                    )

                if show_gradcam:
                    target_layer = get_gradcam_target_layer(model, model_name)
                    gradcam = GradCAM(model, target_layer)
                    try:
                        tensor_grad = transform(image).unsqueeze(0).to(device)
                        heatmap = gradcam.generate(tensor_grad, class_idx=top_idx)
                        overlay = overlay_heatmap(image, heatmap)
                        st.subheader("Grad-CAM (model attention)")
                        st.image(overlay, caption="Regions influencing the prediction", use_container_width=True)
                    finally:
                        gradcam.close()

with tab_about:
    st.markdown(
        """
        **NeuroScan** classifies brain MRI scans into four categories:
        glioma, meningioma, pituitary tumor, or no tumor.

        **Deep learning stack**
        - Fine-tuned **EfficientNet-B0** (or ResNet-18) with transfer learning
        - Data augmentation, early stopping, learning-rate scheduling
        - **Grad-CAM** saliency maps for interpretability

        **Dataset:** [Brain Tumor MRI (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

        > This is a research/education demo — not for clinical diagnosis.
        """
    )
