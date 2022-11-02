import pandas as pd
from collections import defaultdict
import streamlit as st
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from tqdm.notebook import tqdm
tqdm.pandas()


import re
import string
import emoji
import wordninja

import nltk
# import ssl
# https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
'''

#nltk.download("stopwords")
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('english'))
#from api import sm_nlp

import logging

#from transformers import BertTokenizer


def clean_text(text):
    '''
    clean_text:
    :param text str:
    :return: text - 
    '''
    if type(text) == float:
        return text
    # demojize
    text = emoji.demojize(text,delimiters=("", ""))
     
    # remove stopwords and links
    #text = " ".join([token.text for token in sm_nlp(text) if token.text.lower() in stop_words\
    #                 and not token.text.startswith("https://")])
       
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # segment words joined together  
    #text = " ".join(wordninja.split(text))   
    
    # removing specific tag
    text = text.replace("<br />","")
    return text



#@task(name="clean_df")
def clean_df(df):
    '''
    clean_df: assumes that all text-based features are before the
    target variable (last column)
    '''

    for feature in df.columns[:-1]:
        df[feature] = df[feature].progress_apply(clean_text)
    return df


def get_gains_per_feature():
    '''
    get_gains_per_feature: 
    '''
    pass


#@flow(name="feature_allocation",
#      task_runner=SequentialTaskRunner())
def allocate_features(dfs,df_names):
    '''
    allocate_features assumes the df_names and dfs have
    the same ordering of dataframes.
    For files ending in _{number}, group those together
    
    :return: pd.DataFrame - we return a dataframe containing the allocation of
    each feature to a resource.
    '''
    final_dfs = [(clean_df(df),name) for (df, name) in zip(dfs,df_names)]
    st.write("Finished cleaning text...")
    allocation_dict = defaultdict(list)
    for df,name in final_dfs:
        curr_name_tokens = name.split(".")[-2].split("_")
        if curr_name_tokens[-1].isdigit():
            allocation_dict["_".join(curr_name_tokens[:-1])].append(df)
    
    return allocation_dict


