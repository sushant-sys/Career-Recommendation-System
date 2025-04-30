# Career-Recommendation-System
This project is a machine learning–based career recommendation system built using Streamlit, designed to help users explore suitable career paths based on their skills or resume content. 
It uses natural language processing (NLP) and cosine similarity to match user input with predefined career profiles.
How It Works:
The user uploads their resume in PDF format.

The app extracts text using PyMuPDF (fitz).

It compares the resume's content with predefined career skill sets using TF-IDF and Cosine Similarity.

The app then recommends the best matching career.

Features:
Resume text extraction

Machine learning-based skill comparison

Clean and interactive Streamlit interface

Careers included: Data Scientist, Web Developer, ML Engineer, UI/UX Designer, Cybersecurity Analyst

Technologies Used:
Streamlit

PyMuPDF

scikit-learn

pandas
