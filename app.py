import streamlit as st
import os
os.system("python -m spacy download en_core_web_sm")
import api

def landing_page():
    st.write("# Welcome to the ML model preparation tool!")
    st.markdown(
        """
        Data Labs' NLP toolkit has two features: embedding evaluation 
        and load balancing on optimally storing large data for ML models.
        """
    )

if __name__ == "__main__":
    landing_page()
