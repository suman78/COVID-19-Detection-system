import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Constants
IMG_SIZE = (224,224)
MODEL_PATH = "covid_model_mobilenetv2.h5"
TEST_DIR = "Covid19-dataset/test"

# ✅ Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ✅ Title
st.title("🦠 COVID-19 Image Classification System")
st.subheader("Upload an image to detect disease and show model evaluation")

# ✅ Image upload
uploaded_file = st.file_uploader("Upload an Image (X-ray, CT, etc.)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    resized = cv2.resize(image_rgb, IMG_SIZE)
    img_array = np.expand_dims(resized / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Get class labels
    class_labels = list(os.listdir(TEST_DIR))
    class_labels.sort()  # Ensure consistent order

    st.success(f"✅ Predicted Class: **{class_labels[predicted_class]}**")

# ✅ Evaluate model on test set
st.markdown("---")
st.subheader("📊 Model Performance on Test Dataset")

# Load test data
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predictions
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

# Labels
labels = list(test_data.class_indices.keys())

# Metrics
acc = accuracy_score(y_true, y_pred_classes)
cm = confusion_matrix(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes, target_names=labels, output_dict=True)

# Show metrics
st.write(f"**Accuracy:** {acc:.2f}")
st.write("**Precision, Recall & F1-Score per class:**")
st.dataframe({
    label: {
        'Precision': f"{report[label]['precision']:.2f}",
        'Recall': f"{report[label]['recall']:.2f}",
        'F1-Score': f"{report[label]['f1-score']:.2f}"
    }
    for label in labels
})

# Show confusion matrix
st.write("**Confusion Matrix:**")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

