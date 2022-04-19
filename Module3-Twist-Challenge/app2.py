
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime as dt
import geopandas
import plotly.express as px 
import seaborn as sns
import os
import streamlit as st
from streamlit import components
sns.set_theme()

import re
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx
from pyvis.network import Network

import matplotlib.pyplot as plt
from tqdm import tqdm
import boto3
from io import StringIO

## Switch off warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

## Page title
st.title('Armed Conflict Location & Event Network Analysis')

@st.cache
def load_data():
  
  ## Read in data
  aws_access_key_id = 'AKIAZWWRP6SFBRL23XXR'
  aws_secret_access_key = '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'

  client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
          aws_secret_access_key=aws_secret_access_key)

  bucket_name = 'dsi-acled-data'

  object_key = '2019-04-09-2022-04-14.csv'
  csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
  body = csv_obj['Body']
  csv_string = body.read().decode('utf-8')

  acled_data = pd.read_csv(StringIO(csv_string))

  acled_data['date'] = pd.to_datetime(acled_data['event_date'])
  acled_data['year_mon'] = acled_data['date'].dt.to_period('M')
  acled_data['year_mon2'] = acled_data['year_mon'].astype(str)

  ## Replace DRC name
  acled_data['country'] = acled_data['country'].str.replace('Democratic Republic of Congo','DRC')

  return acled_data

acled_data = load_data()

## Filter for Africa
africa = acled_data[acled_data['region'].str.contains("Africa")]

## Filter for country
africa_select = africa[['event_date','year','event_type','actor1','actor2','region','country','notes','fatalities']]

st.write('Select country of interest for network analysis using knowledge graph')

selected_country = st.selectbox(
    'Select Country for Network Analysis',
    africa['country'].unique())

country_data = africa_select[africa_select['country']==selected_country]

#Add title and subtitle to the main interface of the app


st.subheader("Country Network Analysis")
st.write('To generate entities and relations from the ACLED data click the button below.')

if st.button('Generate Entities and Relations'):
  with st.spinner("Generating Entities and Relations"):
    ## Entities Extraction

    def get_entities(sent):

      ## chunk 1
      ent1 = ""
      ent2 = ""

      prv_tok_dep = ""    # dependency tag of previous token in the sentence
      prv_tok_text = ""   # previous token in the sentence

      prefix = ""
      modifier = ""

      #############################################################
      
      for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
          # check: token is a compound word or not
          if tok.dep_ == "compound":
            prefix = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
              prefix = prv_tok_text + " "+ tok.text
          
          # check: token is a modifier or not
          if tok.dep_.endswith("mod") == True:
            modifier = tok.text
            # if the previous word was also a 'compound' then add the current word to it
            if prv_tok_dep == "compound":
              modifier = prv_tok_text + " "+ tok.text
          
          ## chunk 3
          if tok.dep_.find("subj") == True:
            ent1 = modifier +" "+ prefix + " "+ tok.text
            prefix = ""
            modifier = ""
            prv_tok_dep = ""
            prv_tok_text = ""      

          ## chunk 4
          if tok.dep_.find("obj") == True:
            ent2 = modifier +" "+ prefix +" "+ tok.text
            
          ## chunk 5  
          # update variables
          prv_tok_dep = tok.dep_
          prv_tok_text = tok.text
      #############################################################

      return [ent1.strip(), ent2.strip()]

    entity_pairs = []

    for i in tqdm(country_data["notes"]):
      entity_pairs.append(get_entities(i))

    
    def get_relation(sent):
      
      doc = nlp(sent)

      # Matcher class object 
      matcher = Matcher(nlp.vocab)

      #define the pattern 
      pattern = [{'DEP':'ROOT'}, 
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"},  
                {'POS':'ADJ','OP':"?"}] 

      matcher.add("matching_1", None, pattern) 

      matches = matcher(doc)
      k = len(matches) - 1

      span = doc[matches[k][1]:matches[k][2]] 

      return(span.text)


    relations = [get_relation(i) for i in tqdm(country_data["notes"])]


    # extract subject
    source = [i[0] for i in entity_pairs]

    # extract object
    target = [i[1] for i in entity_pairs]

    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

    st.dataframe(kg_df)

    st.write('The top 50 relations discovered are:')

    st.write(pd.Series(relations).value_counts()[:50])

    st.success('Network analysis completed.')

    st.subheader("Relation Knowledge Graph")

    st.write('Select relation to be displayed on the knowledge graph')

    select_relations = pd.Series(relations).value_counts()[:50].index.tolist()

    select_relations = pd.DataFrame(select_relations,columns=['relations'])

    select_relations.to_csv('select_relations.csv',index=False)
    #st.dataframe(select_relations)

    kg_df.to_csv('kg_df.csv',index= False)
select_relations = pd.read_csv('select_relations.csv')
selected_relation = st.selectbox('Select relation to be displayed on knowledge graph',select_relations['relations'])

if st.button('Generate Knowledge Graph'):
  with st.spinner("Generating Knowledge Graph"):
    
    kg_df = pd.read_csv('kg_df.csv')
    #select_relations = pd.read_csv('select_relations.csv')


    G = nx.from_pandas_edgelist(kg_df[kg_df['edge']==selected_relation], "source", "target", 
                              edge_attr=True,create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
    ax = nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
      
    plt.savefig("graph.png")
    st.image('graph.png')

  
