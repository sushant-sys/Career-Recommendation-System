import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Extract text from uploaded resume ===
def extract_text_from_pdf(uploaded_file):
    text = ''
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

# === Sample dataset of careers ===
data = {
    'Career': ['Data Scientist', 'Web Developer', 'ML Engineer', 'UI/UX Designer', 'Cybersecurity Analyst'],
    'Skills': [
        'statistics machine learning python data analysis pandas numpy',
        'html css javascript react node frontend backend',
        'python machine learning deep learning neural networks keras pytorch',
        'design creativity figma adobe xd user experience wireframes',
        'network security encryption cyber attacks threat analysis firewalls'
    ]
}
df = pd.DataFrame(data)

# === Streamlit App ===
st.title("📄 Career Recommendation System")
st.write("Upload your resume (PDF), and we’ll suggest a suitable career path for you!")

uploaded_file = st.file_uploader("Choose your resume (PDF)", type="pdf")

if uploaded_file is not None:
    user_profile = extract_text_from_pdf(uploaded_file)

    vectorizer = TfidfVectorizer()
    career_vectors = vectorizer.fit_transform(df['Skills'])
    user_vector = vectorizer.transform([user_profile])

    similarity = cosine_similarity(user_vector, career_vectors)
    recommended_index = similarity.argmax()
    recommended_career = df.iloc[recommended_index]['Career']

    st.success(f"🔍 Based on your resume, we recommend: **{recommended_career}**")
