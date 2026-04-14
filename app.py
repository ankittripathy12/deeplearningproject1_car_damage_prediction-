import streamlit as st
from model_helper import predict
from PIL import Image

st.title("🚗 Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded File", use_container_width=True)

    # 🔥 Directly pass uploaded_file (NO temp file)
    prediction, confidence = predict(uploaded_file)

    # Output
    if prediction == "Invalid Image":
        st.error("❌ Please upload a valid image.")
    else:
        st.success(f"🚗 Predicted Class: {prediction}")
        st.metric("Confidence", f"{confidence*100:.2f}%")
