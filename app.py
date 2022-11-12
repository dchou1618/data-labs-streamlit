import streamlit as st
import base64
import pymongo
import certifi
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamlit_tags import st_tags

import api

def extract_files(file_type):
    fs = st.file_uploader("Upload file with features", accept_multiple_files=True)
    dfs = []
    df_names = []
    for file in fs:
        try:
            if file_type == "XLSX":
                dfs.append(pd.read_excel(file, "None"))
                df_names.append(file.name)
            else:
                dfs.append(pd.read_csv(file))
                df_names.append(file.name)
        except Exception as e:
            print(e)
            st.warning("Invalid File Type")
            return None, None
    return dfs, df_names


def get_dimension_barplot(dim,list_of_embeddings,names, dim_names):
    fig = go.Figure()  
    for i,embedding in enumerate(list_of_embeddings):
        fig.add_trace(go.Bar(x=dim_names, y=embedding,
                             name=names[i]))
    return fig


def landing_page():
    st.write("# ML model preparation tool")
    st.markdown(
        """
        Data Labs' NLP toolkit has two features: embedding evaluation 
        and load balancing on optimally storing large data for ML models.
        """
    )
    feature = st.selectbox("What toolkit feature do you want to try out?",
              ("Embedding Evaluation","Polar Opposites Word Senses"))
    if feature == "Embedding Evaluation":
        p = base64.b64decode("YnJkMzgyMjM=").decode("utf-8")
        client_url = f"mongodb+srv://dchou_admin:{p}@cluster0.4l7x9tz.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(client_url,
                          tlsCAFile=certifi.where())
        database = client['test']
        model_url = st.selectbox("Huggingface Model",
("Select a Model","allenai/specter","xlnet-base-cased",\
        "bert-base-cased", "roberta-large","facebook/bart-large"))
        if model_url != "Select a Model":
            document_name = st.selectbox("Retrieve personal documents from MongoDB?", ("No",))
            if document_name == "No":
                with st.form("run_get_embeddings"):
                    
                    text_to_test_embeddings = st.text_area("Text to Test "+\
                                   "Feature Extraction Embeddings",
                                   '''Corinna was an ancient Greek lyric poet from Tanagra in Boeotia''')
                    
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
        model_url = st.selectbox("Huggingface Model - Note: size may crash the site given existing trial",
("Select a Model","allenai/specter","xlnet-base-cased",\
        "bert-base-cased", "roberta-large","facebook/bart-large"))
        token_polars = st_tags(label='# Enter first keyword',
                         text='Enter one keyword.',\
                       value=["scientific/religious","hard/soft","rivalry/friendship",
                       "war/love","head/foot","mind/heart","ignite/extinguish"])
        if model_url != "Select a Model" and len(token_polars) > 0:
            with st.form("polar opposite"):
                text1 = st.text_area(f"Text for first keywords","""His scientific jubilee was celebrated in Paris in 1901. All scientific material from the past is making its way online. Computers can connect to and control highly specialized scientific instruments, and equipment can be accessed remotely. Something that is hard is very firm and stiff to touch and is not easily bent, cut, or broken. He shuffled his feet on the hard wooden floor. I've never seen Terry laugh so hard. Old rivalry is no good while one side is sick. The comments will certainly crank up the already intense rivalry between the two teams. But a bitter rivalry soon emerges between the two women. They fought a war over the disputed territory. A war broke out when the colonists demanded their independence. The two countries are at war. She patted the dog on the head. He nodded his head in agreement. The ceiling's low, so watch your head! He read great literature to develop his mind. It's important to keep your mind active as you grow older. He went for a walk to help clear his mind. The fire was ignite by sparks. The paper will ignite on contact with sparks. When he climbs on the deck, his cigarette will ignite the gas, creating a fireball. """)
                text2 = st.text_area(f"Text for polar opposite words","""My religious beliefs forbid the drinking of alcohol. Religious leaders called for an end to the violence. His wife is very active in the church, but he's not religious himself. Regular use of a body lotion will keep the skin soft and supple. When it's dry, brush the hair using a soft, nylon baby brush. This pillow is too soft for me. They have enjoyed many years of friendship. He was encouraged by the friendship his coworkers showed him. Friendship helps ambitions and dreams you share turn into positive action. I love the fact that you always seem to care so much. She fell in love with him the first time she met him. To love and to be loved is the greatest happiness. She stamped her foot again. David called to the children from the foot of the stairs. A single word at the foot of a page caught her eye. I could feel my heart pounding. He has a bad heart. He put his hand on his heart. The fire department was called in to extinguish the blaze. He will extinguish his cigarette in the ashtray. When they ruthlessly extinguish all resistance, peace will be restored. News of the conflict will extinguish our hopes for a peaceful resolution. """)
                clicked = st.form_submit_button("Verify Polar Opposite Interpretability in Vector Dimensions")
                st.write("""Currently, we support visualizing vector differences between polar opposites. We will add
orthogonality maximization to obtain better polar embeddings for individual tokens.""")
                if len(text1) > 0 and len(text2) > 0 and clicked:
                    output, names, dim_names = api.compare_polar_opposites(token_polars, text1, text2, model_url)
                    fig = get_dimension_barplot(len(output[0]),output, names, dim_names)
                    fig.update_layout(title=f"Polar Opposite-Adjusted Embeddings")
                    st.plotly_chart(fig, use_container_width=True)
                    

if __name__ == "__main__":
    landing_page()
