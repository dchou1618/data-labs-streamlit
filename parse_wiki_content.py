#!/usr/bin/env python3
# encoding: utf-8
'''
brief: parse_wiki_content.py takes the raw wikipedia
content returned from api.py get_embeddings.

author: Dylan V. Chou
'''

import re
import api
from num2words import num2words
from english_words import english_words_set as word_set


def parse(raw_content, title):
    # Transform individual numbers without decimals 
    # (inclusion of "point" may impact embeddings).
    
    transform_individual_regex = re.compile(r"($[0-9][^0-9]|[^0-9][0-9][^0-9])")
    raw_content = str(raw_content).replace("\n"," ")
    fin_str = ""
    last_end = 0
    for m in transform_individual_regex.finditer(raw_content):
        start = m.start()
        end = m.end()
        prec, after = start+1,end-1
        if (end-start == 2):
            if (raw_content[start].isdigit()):
                digit_str = raw_content[start:end-1]
                prec = start
            else:
                digit_str = raw_content[start+1:end]
                after = end
        else:
            assert end-start == 3, "Not three characters long (single digit surrounded by non-digits)"
            digit_str = raw_content[start+1:end-1]
        fin_str += raw_content[last_end:prec] +\
                      num2words(int(digit_str)) + raw_content[after:end]
        last_end = end
    fin_str += raw_content[last_end:]
    fin_str = fin_str
    

    sections = fin_str.split("==")
    sentences_of_interest = []
    for section in sections:
        if len(section) == 0: continue
        new_section = ("" if section[0] == "=" else section[0]) +\
    	section[1:-1] +\
    	("" if section[-1] == "=" else section[-1])
        total_sentences = 0
        sentences = []
        non_existent_word = False
        for sent in api.sm_nlp(new_section).sents:
            for token in sent:
                str_token = str(token)
                if str_token not in word_set and str_token[0].islower() and\
		   str_token[-1] != "." and str_token[0] != ".":
                    non_existent_word = True
                    break
            if non_existent_word:
                break
            str_sent = str(sent)
            if re.search(f"([^A-Za-z]|){title}([^A-Za-z]|)",str_sent.lower()):
                sentences.append(str_sent.strip())
            total_sentences += 1
        if total_sentences > 1:
            sentences_of_interest += sentences
    return sentences_of_interest
