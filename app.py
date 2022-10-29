import streamlit as st
import base64
import pymongo
import certifi
import pandas as pd
import plotly.express as px
import api

def extract_files(file_type):
    fs = st.file_uploader("Upload file with features", accept_multiple_files=True)
    dfs = []
    for file in fs:
        try:
            if file_type == "XLSX":
                dfs.append(pd.read_excel(file, "None"))
            else:
                dfs.append(pd.read_csv(file))
        except Exception as e:
            print(e)
            st.warning("Invalid File Type")
            return None
    return dfs


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
                with st.form("run_get_embeddings"):
                    
                    text_to_test_embeddings = st.text_area("Text to Test "+\
                                   "Feature Extraction Embeddings",
                                   '''''')
                    
                    clicked = st.form_submit_button("Run Word Sense Disambiguation Evaluation Over Embeddings")
                    if len(text_to_test_embeddings) > 0 and len(model_url) > 0 and clicked:
                        output = api.get_embeddings(None,model_url, database, description=text_to_test_embeddings)
                        df = {"dim1":[],"dim2":[],"dim3":[],"label":[],"context":[]}
                        for word in output["relevant_lists_wsd"]:
                            for vect in output["relevant_lists_wsd"][word]:  
                                 for index, reduced_vect in vect["reduced_context_embeddings"]:
                                     df["dim1"].append(reduced_vect[0])
                                     df["dim2"].append(reduced_vect[1])
                                     df["dim3"].append(reduced_vect[2])
                                     df["label"].append(word.lower())
                                     df["context"].append(vect["tokenized_passage"])
                        df = pd.DataFrame(df)
                        fig = px.scatter_3d(df, x="dim1",y="dim2",z="dim3",color="label")
                        st.plotly_chart(fig, use_container_width=True)
                        st.json(output)
            else:
                pass
    else:
        file_type = st.selectbox("Data File Type", ("Select a type","CSV","XLSX"))
        if file_type != "Select a type":
            dfs = extract_files(file_type)
            if dfs is not None:
                for df in dfs:
                    st.write(df)
            
        

if __name__ == "__main__":
    landing_page()
