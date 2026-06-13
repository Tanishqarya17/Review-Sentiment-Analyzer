import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Load models and vectorizer
# ---------------------------
models = {
    "Logistic Regression": joblib.load("sentiment_model_lr.pkl"),
    "Linear SVM": joblib.load("sentiment_model_svm.pkl"),
    "Random Forest": joblib.load("sentiment_model_rf.pkl")
}

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Sentiment Analyzer 💬",
    page_icon="💬",
    layout="centered"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #F8F9FB;
}
.main {
    background: #FFFFFF;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.1);
}
h1 {
    color: #1E88E5;
    text-align: center;
}
.stButton button {
    background-color: #1E88E5;
    color: white;
    font-size: 1.1em;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stButton button:hover {
    background-color: #1565C0;
}
.sentiment-box {
    text-align: center;
    padding: 1.5em;
    border-radius: 15px;
    margin-top: 1.5em;
    font-size: 1.3em;
}
.positive {
    background-color: #E3FCEF;
    color: #2E7D32;
    border: 2px solid #A5D6A7;
}
.negative {
    background-color: #FFEAEA;
    color: #C62828;
    border: 2px solid #EF9A9A;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# App Header
# ---------------------------
st.title("💬 Sentiment Analysis Dashboard")
st.write("Analyze the sentiment of customer reviews using trained ML models. Choose a model, type a review, and view the result instantly.")

# ---------------------------
# Model Selection
# ---------------------------
model_choice = st.selectbox(
    "🔍 Choose your model:",
    list(models.keys())
)

# ---------------------------
# Input Box
# ---------------------------
review_text = st.text_area(
    "📝 Enter a review below:",
    placeholder="Type or paste a review here (e.g., 'The product quality is amazing!')"
)

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("🚀 Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        # Vectorize input
        X_vec = vectorizer.transform([review_text])

        # Select model and predict
        model = models[model_choice]
        prediction = model.predict(X_vec)[0]

        # Try to get prediction probabilities
        prob = None
        try:
            prob = model.predict_proba(X_vec)[0]
        except Exception:
            # SVM doesn’t have predict_proba
            prob = np.array([1 - prediction, prediction])

        # Display result
        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        confidence = max(prob) * 100

        color_class = "positive" if prediction == 1 else "negative"
        st.markdown(f'<div class="sentiment-box {color_class}">\
                        <strong>Predicted Sentiment:</strong> {sentiment}\
                      </div>', unsafe_allow_html=True)

        # Confidence bar
        st.progress(int(confidence))
        st.write(f"**Model Confidence:** {confidence:.2f}%")

        # Model info
        st.info(f"Model used: **{model_choice}**")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Built with ❤️ using Streamlit, scikit-learn & Python</p>",
    unsafe_allow_html=True
)
