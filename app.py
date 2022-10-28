import streamlit as st
import base64
import pymongo
import certifi
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
        p = base64.b64decode("YnJkMzgyMjM=").decode("utf-8")
        client_url = f"mongodb+srv://dchou_admin:{p}@cluster0.4l7x9tz.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(client_url,
                          tlsCAFile=certifi.where())
        database = client['test']
        model_url = st.selectbox("Huggingface Model",
("Select a Model","allenai/specter","xlnet-base-cased",\
        "bert-base-cased", "roberta-large","facebook/contriever-msmarco"))
        if model_url != "Select a Model":
            document_name = st.selectbox("Retrieve personal documents from MongoDB?", ("Login", "None"))
            if document_name == "None":
                text_to_test_embeddings = st.text_area("Text to Test "+\
                                   "Feature Extraction Embeddings",
                                   '''''')
                if len(text_to_test_embeddings) > 0 and len(model_url) > 0:
                    st.json(api.get_embeddings(None,model_url, database, description=text_to_test_embeddings))
                    text_to_test_embeddings = ""
            else:
                pass
    else:
        pass

if __name__ == "__main__":
    landing_page()
