#!/usr/bin/env python3
# encoding: utf-8

'''
brief: api.py is a script meant to house the API endpoints
of our "user recommendations" project. We test the 
correctness of the API endpoint behavior using pytest. The 
API will be deployed to a Heroku website.

motivation: While not novel, the project is meant to implement recent 
papers on word sense disambiguation and traditional 
recommendation systems (collaborative filtering, multi-armed 
bandits).

author: Dylan V. Chou
'''
# nlp
from transformers import pipeline
import spacy
import wikipedia
import parse_wiki_content
from transformers import BertTokenizer, BertModel, AutoTokenizer

# general
import json
import base64
from bson import json_util
import numpy as np
import requests
from tqdm import tqdm
from dotenv import dotenv_values
from transformers import AutoTokenizer

# MongoDB & Flask application
import pymongo
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, reqparse, abort, fields, marshal_with, Resource
from flask_mongoengine import MongoEngine
from flask_graphql import GraphQLView
import certifi
import pandas as pd

# Schema definitions
import definitions


import streamlit as st

num_summaries = 0

# load the pipeline once
@st.cache(allow_output_mutation=True)
def load_en_core_sm(name):
    sm_nlp = spacy.load(name)
    return sm_nlp

sm_nlp = load_en_core_sm("en_core_web_sm")

entities = set(sm_nlp.get_pipe("ner").labels)
#app = Flask(__name__)

#search_client = search_data_labs.Search(config["CLOUD_ID"], config["ELASTIC_PASSWORD"])

#txt_to_embeddings = pipeline('feature-extraction', model="bert-base-uncased")
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#print(tokenizer.tokenize("2 (two) is a number, numeral and digit."))
#print(np.array(txt_to_embeddings("This is a sentence.\n\nWhy are we here?")[0]).shape)

'''
Definition of a product -- has a name, price, description and
'''

class Product(Resource):
    def __init__(self,name,desc,_id,price,weight=None):
        self.name = name
        self.desc = desc
        self._id = _id
        self.price = price
        self.weight = weight

# curl -H "Content-Type: application/json" -X GET "http://127.0.0.1:5000/?name=sue"
# argument is typically ?name=sue.

'''
GET methods:
'''

'''
:brief: get_recommended - obtains the recommended products to a given product
- we have the paradigm of a multi-armed bandit, where each arm corresponds to 
an unknown probability distribution.
:return:
'''

'''
@app.route('/get_recommended/<name>/<_id>/', methods=["GET"])
def get_recommended(name, id):
    return
'''

def print_array(arr):
    if type(arr) == np.array:
        print(arr.shape)
    elif type(arr) == list:
        print("Length: ",len(arr))
        if len(arr) == 1:
            print(len(arr[0]))

def return_vect(vect):
    return vect[-1] if len(vect) > 0 else []

'''
:brief: embedding_dim_reduction - based on Achlioptas, the use of 
a randomly generated matrix formed by sqrt(3), 0, or -sqrt(3) is 
more computationally efficient than gaussian random projection.
:param: token_dict - a dictionary mapping a token
to a list of tuples containing lists of embeddings 
of length equal to the number of occurrences of a word in context.
:return: dictionary mapping token to contextual embeddings.
We evaluate quality of embeddings in terms of how separate 
the .
'''
def embedding_dim_reduction(token_dict):
    overall_vectors = []
    total_length = 0
    for token in token_dict:
	# extracting second entries from embeddings.
        for i in range(len(token_dict[token])):            
            total_length += len(token_dict[token][i][-1])
            overall_vectors += list(map(lambda v: list(v[1]), token_dict[token][i][-1]))
    # overall_vectors contains all the embeddings of the words in relevant contexts. Reshape
    # into dxN (d dimensions, N observations) matrix and multiply with random matrix R
    # containing a random distribution of sqrt(3)*(0 2/3 prob, 1 1/6 prob, -1 1/6 prob).
    flat_overall = np.array(list(map(lambda x: x[0], overall_vectors)) ).flatten()

    M = len(overall_vectors[0])
    N = flat_overall.shape[0]

    X = np.reshape(np.array(overall_vectors), (M,N))
    R = np.random.choice(a=[np.sqrt(3), 0, -1*np.sqrt(3)], 
    			 size=(M//256,M), 
                         replace=True, p=[2/3,1/6,1/6])
    res = np.reshape(np.matmul(R,X), (N,M//256))
    # N 6-length vectors
    reduced_vectors = dict()
    # i - tracks the current index in
    # the set of all embedding arrays for each respective token.
    i = 0
    for token in token_dict:
        token_contents = []
        for relevant_contents in token_dict[token]:
            token_content = dict()
            num_embeds = len(relevant_contents[-1])

            token_content["reduced_context_embeddings"] = list(zip(list(map(lambda x: x[0], relevant_contents[-1])),
                                                               res[i:(i+num_embeds)]))
            token_content["original_title"] = relevant_contents[1]
            token_content["word_of_interest"] = relevant_contents[2]
            token_content["tokenized_passage"] = relevant_contents[4]
            token_contents.append(token_content)
            i += num_embeds
        reduced_vectors[token] = token_contents
    
    return reduced_vectors

 


def iterate_over_wiki_instances(title_embedding,original_title, word, max_num, seen_pages, args, auto_suggest=False):
    '''
    iterate_over_wiki_instances goes over all available contexts of a word 
    and obtains their embeddings. Per embedding, it's composed of N+2 embedding
    vectors for text with N words. The 2 is added to account for start and 
    end tokens: <s> and </s>.
    '''
    summaries = []

    global num_summaries
    if (num_summaries > max_num or word in seen_pages or len(seen_pages) > 50):
        return []
    seen_pages.add(word)
    try:
        #curr_page = wikipedia.page(word, auto_suggest=auto_suggest)
        #content = parse_wiki_content.parse(curr_page.content, word.lower())
        using_lemma = False
        try:
            content = wikipedia.summary(sm_nlp(word)[0].lemma_, sentences=5).split("\n")
            using_lemma = True
        except:
            content = wikipedia.summary(word, sentences=5).split("\n")

        '''
        print("##############")
        print(f"Word: {word}, {sm_nlp(word)[0].lemma_}",
              wikipedia.summary(sm_nlp(word)[0].lemma_, sentences=5).split("\n"))
        print("-----------------")
        print(content)
        print("##############")
        ''' 
        for passage in content:
            if "api_url" in args:
                resp = query_model_id(passage, args["api_url"], args["headers"])
            else:
                resp = args["txt_to_embeddings"](passage)
            print_array(resp)
            passage_embedding = np.array(resp[0])
            tokenized_passage = sm_nlp(passage)
            passage_embedding = average_pooling(tokenized_passage, passage_embedding, args["tokenizer"])
            # tokenized_passage = args["tokenizer"].tokenize(passage)
            # tuples of (token_position, embedding)
            token_embeddings = []
            i = 0
            for token in tokenized_passage:
                lower_token = str(token).lower()
                lower_original = original_title.lower()
                if ((sm_nlp(lower_token)[0].lemma_ == sm_nlp(lower_original)[0].lemma_) if\
                    using_lemma else lower_token == lower_original):
                    #print("Passage Embedding",passage,token,passage_embedding[i][:20])
                    #print("Found token position for",str(token))
                    if not any(pd.isna(x) for x in passage_embedding[i]):
                        token_embeddings.append((i,passage_embedding[i]))
                i += 1
            if len(token_embeddings) > 0:
                summaries.append([title_embedding,original_title, word, [passage],
                                     [token.text for token in tokenized_passage], token_embeddings])
    
    # wikipedia.exceptions.DisambiguationError
    except Exception as e:
        print("Error: ",e,original_title, word, sm_nlp(word)[0].lemma_)
        summaries = []
        try: 
            for title in e.options:
                summaries += iterate_over_wiki_instances(title_embedding, original_title, title, max_num, seen_pages, args)
                
        except Exception as e1:
            print("Last exception:",e1)
            pass
    num_summaries += len(summaries)
    return summaries

def query_model_id(text, api_url, headers):
    r = requests.post(api_url, headers=headers, json={"inputs": text, "options":{"wait_for_model":True}})
    return r.json()


def average_pooling(doc, token_embeddings, tokenizer):
    '''
    average_pooling - applies mean pooling to multiple BPE
    embeddings associated with the same token.
    :param doc spacy.Doc: the document (list of sentences)
    :param tokenizer transformers.models: 
    :return: 
    '''
    embeddings = []
    tokenizer_bpe = tokenizer(doc.text,
                        return_offsets_mapping=True)["offset_mapping"]
    i = 0
    for token in doc:
        start = token.idx
        end = start+len(token.text)
        if i == len(tokenizer_bpe):
            break
        curr_range = tokenizer_bpe[i]
        first_embedding = i
        while i < len(tokenizer_bpe) and curr_range[1] <= end:
            curr_range = tokenizer_bpe[i]
            # print("current range and start/end index: ",curr_range,start,end)
            i+=1
            if curr_range[1] == curr_range[0]:
                first_embedding = i
        i-=1
        last_embedding = i
        curr_embeddings = token_embeddings[first_embedding:last_embedding]
        # print("Pooling indices: ", token, len(curr_embeddings), first_embedding, last_embedding)
        mean_pooled_embedding = np.sum(curr_embeddings,axis=0)/len(curr_embeddings)
        embeddings.append(mean_pooled_embedding)
        
    return embeddings

def run_embeddings_for_word(txt_to_embeddings, tokenizer, texts, token1):
    doc = sm_nlp(texts)
    token_embedding = txt_to_embeddings(doc.text)[0]
    token_embedding = average_pooling(doc, token_embedding, tokenizer)
    token_embeddings = []    
    for sent in doc.sents:
        for token in sent:    
            if token1.lower() == token.text.lower():
                token_embeddings.append(token_embedding[token.i])
                break
    return token_embeddings

#https://dl.acm.org/doi/abs/10.1145/3366423.3380227?casa_token=5IHmDnDhgX0AAAAA:GiKepf7xwlKswTVo_fFthCBQCkWZVy8BMUthBdeDZmbo0Y4gbXnzqQSw-UP77p1esGM9G-IJ_hBRpg
def compare_polar_opposites(token1, texts1, token2, texts2, model_id):
    '''
    compare_polar_opposite - based on the above paper, we use semantic differentials from
    Osgood, Charles Egerton, George J. Suci, and Percy H. Tannenbaum. The measurement of meaning. No. 47. University of Illinois press, 1957.
    
    The function is intended to verify whether non-negative dimensions of one vector is associated with one topic. 
    :param texts1 str:
    :param texts2 str:
    :return:
    '''
    token1 = token1.lower()
    token2 = token2.lower()
    
    txt_to_embeddings = pipeline("feature-extraction", model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)    

    token_embeddings1 = run_embeddings_for_word(txt_to_embeddings, tokenizer, texts1, token1)
    token_embeddings2 = run_embeddings_for_word(txt_to_embeddings, tokenizer, texts2, token2)
    
    return token_embeddings1, token_embeddings2


'''
:brief: get_embeddings uses embeddings trained at the same time as 
a neural network was trained. Namely, bert base's uncased model 
has dimensions (2+#number of tokens,768) in its 'feature-extraction' pipeline.
768 is the length of the embedding and the first dimension denotes the 
token. The 2 baseline tokens come from the start and end tokens of a 
sentence <s> </s>. A third dimension is created, but is typically 1 (first dimension).
:param: name - 
:param: _id - 
:return: dictionary of the name, id item entry in the database as
well as the associated embeddings. A disambiguation evaluation
of the embeddings is done to identify whether Euclidean distance between
embeddings is consistent with similarity in the context that the word is being
used in. Future work involves identifying collections of vector entries that 
may correspond to contrasting topics, such as good vs. evil, high vs. low. 
Another route of work can be to handle multi-token titles. One can observe how 
the pretrained tokenizer associated with the model breaks down the title and 
shift a sliding window of that length.

Inputs: feature_extraction, bert-base-uncased.
'''
import logging

def get_embeddings(name,model_id, database, using_api=False, **kwargs):
    # https://huggingface.co/blog/getting-started-with-embeddings
    # Modification where we use huggingface api instead of bert tokenizer.
    usr_collection = database["documents"]
    
    txt_to_embeddings = pipeline("feature-extraction", model=model_id)

    resp = usr_collection.find_one({"name":name})

    if resp is None:
        if "description" in kwargs:
            doc = sm_nlp(kwargs["description"])
        else:
            assert False, "specify description."
    else:
        doc = sm_nlp(resp["description"])
    relevant_lists = dict()
    token_embeddings = txt_to_embeddings(doc.text)[0]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    token_embeddings = average_pooling(doc, token_embeddings, tokenizer)
    #print(token_embeddings)
    # print(len(token_embeddings), [len(embed) for embed in token_embeddings])
    for sent in doc.sents:
        for token in sent:
            #token_embedding = query_model_id(str(token), api_url, headers)
            # print("Document: "+doc.text+"\n\n")
            token_embedding = token_embeddings[token.i]
            #print(token, token.ent_type_, entities, token.pos_)
            if token.ent_type_ in entities or token.pos_ in {"NOUN", "ADJ"}:
                global num_summaries 
                num_summaries = 0
                # print("Token: "+str(token)+"\n\n")
                relevant_contents = iterate_over_wiki_instances(token_embedding, str(token),
                                                                str(token), 20, set(), {"api_url":api_url,\
                                                                "headers":headers,"tokenizer":tokenizer})\
                                    if (bool(using_api)) else\
                                    iterate_over_wiki_instances(token_embedding, str(token),
                                                               str(token), 20, set(), {"tokenizer":tokenizer,\
                                                                "txt_to_embeddings":txt_to_embeddings})
                if len(relevant_contents) > 0:
                    relevant_lists[str(token)] = relevant_contents
    # print(relevant_lists)
    relevant_lists = embedding_dim_reduction(relevant_lists)
    if resp is None:
        resp = {"model_id":model_id,"description":kwargs["description"],
               "description_embeddings":token_embeddings,"relevant_lists_wsd": relevant_lists}
    else:
        resp["relevant_lists_wsd"] = relevant_lists    
    return json.loads(json_util.dumps(resp))



def put_ml_feature(name, memory, gain,compressed_memory,compressed_gain,available_services):
    usr_collection = database["ml"]    
    new_ml_feature = {"name": name, "memory": float(memory),
                      "gain": float(gain),
                      "compressed_memory": float(compressed_memory),
		      "compressed_gain": float(compressed_gain),
                      "avaiable_services": available_services[1:-1].split(",")}
    
    print(usr_collection.insert_one(new_ml_feature))
    return json.loads(json_util.dumps(new_ml_feature))






def post_input_data():
    item_name = request.form['name']
    item_id = request.form["id"]
    res = get_embeddings(item_name, item_id)
    with open("data/raw/output.json", "w", encoding='utf-8') as f:
    	json.dump(res, f, ensure_ascii=False, indent=4)
    f.close()
    #except:
    #    prod_name = request.form["prod_name_text"]
    #    res=get_product(prod_name) 
    #    print(res)
    return render_template("index.html",title="home page")

'''
:brief: GET request for an item name, description and list of items
with the specified name and id.
:param: name - name of the item. For now, the attributes of "items" 
are kept abstract as the project is meant to encompass a variety
of nlp and recommendation applications.
:param: _id - item id.
:return: the json dictionary specifying the name, description, and
the full item entry.
'''

#@app.route('/get_product/<name>/', methods=['GET'])
def get_product(name):
    col = database["users"]
    records = list(col.find({"name":name}))
    assert len(records) == 1, "Not exactly one record. Duplicate records."
    return json.loads(json_util.dumps({"name": name,
    		    "description": records[0]["description"],
		    "list_of_items": records}) )

'''
POST methods:
'''

'''
:brief: adds product to the database using POST method. The name,
description, product id, and price need to be specified.
:param: name - name of the item.
:param: desc - description (at least one character long).
:param: prod_id - id of the .
:param: price - price of the item.
:return: json response of the new user that's added (may lead to 
duplicates in database).
'''

def put_users(name, desc, prod_id, price):
    product_weight = request.args.get("weight")
    has_been_added = False
    usr_collection = database["users"]
    new_user = {"name":name,
    	            "description":desc,
		    "id":int(prod_id),
		    "price":float(price),
		    "weight":float(product_weight)}
    print(usr_collection.insert_one(new_user))
    return json.loads(json_util.dumps(new_user))

def delete_product(name,_id):
    usr_collection = database["users"]
    query_prompt = {"name":name,"id": int(_id)}
    res = usr_collection.delete_one(query_prompt)
    print(f"{res.deleted_count} documents deleted.")
    return json.loads(json_util.dumps(query_prompt))    

'''
:brief: updates item with the specified name and prod_id.
:param: name - name of existing item.
:param: desc - updated description.
:param: prod_id - id of existing item.
:param: price - price of existing item.
:return: json response of the new user being updated.
'''

def update_users(name, desc, prod_id, price):
    product_weight = request.args.get("weight")
    usr_collection = database["users"]
    new_user_to_update = {"name":name,
    			  "description":desc,
			  "id": prod_id,
			  "price": price,
			  "weight": product_weight
			 }
    query = {"name": name, "id": prod_id}
    usr_collection.update_one(query, {"$set":new_user_to_update}, upsert=False)
    return json.loads(json_util.dumps(new_user_to_update))


if __name__ == "__main__":
    pass
    '''   
    p = base64.b64decode("YnJkMzgyMjM=").decode("utf-8")
    client_url = f"mongodb+srv://dchou_admin:{p}@cluster0.4l7x9tz.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(client_url,
                          tlsCAFile=certifi.where())
    database = client['test']
    get_embeddings(None,"allenai/specter", database, description="Lichfield Cathedral, in Lichfield, Staffordshire, is the only medieval English cathedral with three spires.")
    '''
