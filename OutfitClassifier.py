"""Streamlit UI for ClothingClassifier model inference and evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


MODEL_PATH = "best_tuned_vgg16.keras"
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def label_for(index, class_names):
    if class_names and 0 <= int(index) < len(class_names):
        return class_names[int(index)]
    return str(int(index))


def predict_image(model, image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    batch = np.asarray(image, dtype=np.float32)[None, ...]
    output = np.asarray(model.predict(batch, verbose=0))
    if output.ndim > 1 and output.shape[-1] > 1:
        return int(np.argmax(output[0])), float(np.max(output[0]))
    return int(np.rint(output.reshape(-1)[0])), None


def resolve_image_path(raw_path):
    path = Path(str(raw_path).strip()).expanduser()
    candidates = []

    if path.is_absolute():
        candidates.append(path)
        if not path.suffix:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.append(path.with_suffix(ext))
    else:
        candidates.append(Path.cwd() / path)
        candidates.append(Path.cwd() / "images" / path)
        if not path.suffix:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidates.append(Path.cwd() / path.parent / f"{path.name}{ext}")
                candidates.append(Path.cwd() / "images" / f"{path.name}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else path


st.set_page_config(page_title="ClothingClassifier", page_icon="👕", layout="wide")
st.title("ClothingClassifier")
st.write("Upload an image for classification, or evaluate predictions using a test CSV.")

if not Path(MODEL_PATH).is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()

output_shape = model.output_shape
class_count = int(output_shape[-1]) if isinstance(output_shape, tuple) and len(output_shape) > 1 else 1
st.sidebar.header("Class mapping")
class_names = [s.strip() for s in st.sidebar.text_area(
    "Enter category names in model output order, one per line",
    placeholder="T-shirt\nTrouser\nPullover",
).splitlines() if s.strip()]
if class_names and len(class_names) != class_count:
    st.sidebar.warning(f"Model has {class_count} outputs; unmatched indices display as numbers.")

image_tab, csv_tab = st.tabs(["Single image", "Test CSV"])
with image_tab:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            st.image(image, caption=uploaded.name, width=350)
            index, confidence = predict_image(model, image)
            st.success(f"Predicted category: **{label_for(index, class_names)}**")
            if confidence is not None:
                st.write(f"Confidence: {confidence:.2%}")
        except Exception as exc:
            st.error(f"Image prediction failed: {exc}")

with csv_tab:
    st.caption("CSV must contain image paths. Provide an actual-label column to generate evaluation metrics.")
    csv_upload = st.file_uploader("Choose test CSV", type=["csv"], key="csv_upload")
    if csv_upload is not None:
        try:
            data = pd.read_csv(csv_upload)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            st.stop()
        if data.empty or len(data.columns) == 0:
            st.error("The CSV is empty or has no columns.")
            st.stop()
        columns = data.columns.tolist()
        path_col = st.selectbox("Image path column", columns)
        actual_col = st.selectbox("Actual category column (optional)", ["— None —"] + columns)

        if st.button("Run predictions", type="primary"):
            records = []
            progress = st.progress(0.0)
            for i, (_, row) in enumerate(data.iterrows()):
                raw_path = str(row[path_col]).strip()
                path = resolve_image_path(raw_path)
                actual = None if actual_col == "— None —" else str(row[actual_col])
                record = {"image_path": str(path), "actual": actual, "predicted": None, "status": ""}
                try:
                    with Image.open(path) as image:
                        index, confidence = predict_image(model, image)
                    record["predicted"] = label_for(index, class_names)
                    record["status"] = "OK"
                    if confidence is not None:
                        record["confidence"] = confidence
                except Exception as exc:
                    record["status"] = f"Error: {exc}"
                records.append(record)
                progress.progress((i + 1) / len(data))

            results = pd.DataFrame(records)
            st.subheader("Full image-path results: actual vs predicted")
            st.dataframe(results, use_container_width=True)
            st.download_button("Download results CSV", results.to_csv(index=False).encode("utf-8"),
                                file_name="outfit_predictions.csv", mime="text/csv")

            if actual_col != "— None —":
                valid = results[results["status"].eq("OK") & results["actual"].notna()]
                if valid.empty:
                    st.warning("No valid labeled predictions are available for evaluation.")
                else:
                    actual = valid["actual"].astype(str)
                    predicted = valid["predicted"].astype(str)
                    labels = sorted(set(actual) | set(predicted))
                    st.subheader("Classification report")
                    report = classification_report(actual, predicted, labels=labels,
                                                   output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df[["precision", "recall", "f1-score", "support"]],
                                 use_container_width=True)
                    st.subheader("Confusion matrix")
                    matrix = confusion_matrix(actual, predicted, labels=labels)
                    st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels),
                                 use_container_width=True)