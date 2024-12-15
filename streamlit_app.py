import cv2
import numpy as np
import streamlit as st
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# LBP Parameters
RADIUS = 1
POINTS = 8 * RADIUS
model_path = "knn_model.pkl"

# Functions for Preprocessing and Feature Extraction
#def preprocess_image(image_path):
 #   img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #  kernel = np.ones((3, 3), np.uint8)
  #  clean_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#    return clean_img
def preprocess_image(uploaded_file):
    # Read the uploaded image from the buffer
    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully read
    if img is None:
        st.error("Failed to read the image. Please upload a valid image file.")
        return None
    # Apply thresholding and other preprocessing steps
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    clean_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    return clean_img


def extract_lbp_features(image):
    lbp = local_binary_pattern(image, POINTS, RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, POINTS + 3), range=(0, POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Streamlit Application
def main():
    st.title("Fingerprint Comparison System Using LBP & KNN")
    
    # Uploading images
    uploaded_img1 = st.file_uploader("Upload First Fingerprint Image", type=["jpg", "jpeg", "png"])
    uploaded_img2 = st.file_uploader("Upload Second Fingerprint Image", type=["jpg", "jpeg", "png"])

    if uploaded_img1 and uploaded_img2:
        # Convert uploaded files into images
        img1 = cv2.imdecode(np.frombuffer(uploaded_img1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(np.frombuffer(uploaded_img2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Display original images
        st.subheader("Original Fingerprint Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption="First Fingerprint", use_column_width=True)
        with col2:
            st.image(img2, caption="Second Fingerprint", use_column_width=True)

        # Step 1: Preprocess Images
        st.subheader("Preprocessing - Thresholding and Cleaning")
        preprocessed_img1 = preprocess_image(uploaded_img1)
        preprocessed_img2 = preprocess_image(uploaded_img2)
        col1, col2 = st.columns(2)
        with col1:
            st.image(preprocessed_img1, caption="Processed First Fingerprint", use_column_width=True)
        with col2:
            st.image(preprocessed_img2, caption="Processed Second Fingerprint", use_column_width=True)

        # Step 2: Extract LBP Features
        st.subheader("Feature Extraction - Local Binary Pattern (LBP)")
        features1 = extract_lbp_features(preprocessed_img1)
        features2 = extract_lbp_features(preprocessed_img2)
        
        st.write("Extracted Features from First Image:", features1)
        st.write("Extracted Features from Second Image:", features2)
        
        # Train KNN Model
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit([features1, features2], ["person1", "person2"])

        # Save the model (optional)
        joblib.dump(knn, model_path)

        # Step 3: Test the model with both images
        predicted_label1 = knn.predict(features1.reshape(1, -1))
        predicted_label2 = knn.predict(features2.reshape(1, -1))

        # Step 4: Calculate accuracy (for simplicity in this case)
        accuracy = accuracy_score(["person1", "person2"], [predicted_label1[0], predicted_label2[0]]) * 100
        st.subheader(f"Model Accuracy: {accuracy:.2f}%")

        # Step 5: Compare the two fingerprint images
        st.subheader("Fingerprint Comparison Result")
        if predicted_label1 == predicted_label2:
            st.success("Same Person")
        else:
            st.error("Different Person")

        # Additional Visualization (Optional)
        st.subheader("Edge Detection - Canny")
        img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
        img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)

        edges1 = cv2.Canny(img1_blur, 50, 150)
        edges2 = cv2.Canny(img2_blur, 50, 150)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(edges1, cmap='gray')
        ax1.set_title("Fingerprint 1 - Edges")
        ax1.axis('off')
        
        ax2.imshow(edges2, cmap='gray')
        ax2.set_title("Fingerprint 2 - Edges")
        ax2.axis('off')
        st.pyplot(fig)

# Run the Streamlit app
if __name__ == "__main__":
    main()
