import streamlit as st
import api

def landing_page():
    st.write("# ML model preparation tool")
    st.markdown(
        """
        Data Labs' NLP toolkit has two features: embedding evaluation 
        and load balancing on optimally storing large data for ML models.
        """
    )
    feature = st.selectbox("What toolkit feature do you want to try out?",
              ("Embedding Evaluation","Load Balancing Large Data"))
    if feature == "Embedding Evaluation":
        model_url = st.text_input("Huggingface Model URL")
        text_to_test_embeddings = st.text_area("Text to Test "+\
                                   "Feature Extraction Embeddings",
                                   '''''')
    else:
        pass

if __name__ == "__main__":
    landing_page()
