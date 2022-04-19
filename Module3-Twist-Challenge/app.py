
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
from pyvis.network import Network
sns.set_theme()

import re
import bs4
import requests
import spacy
from spacy import displacy
spacy.cli.download("en")

nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm
import boto3
from io import StringIO

## Switch off warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

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

#Add title and subtitle to the main interface of the app
st.title('Armed Conflict Location & Event Data Analysis')

min_date = acled_data['date'].min()
max_date = acled_data['date'].max()

start_date = st.date_input('Start Date', value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input('End Date', value=max_date, min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

acled_data = acled_data[((acled_data['date'] >= start_date) & (acled_data['date'] < end_date))]

st.subheader("Africa Analytics")

## Filter for Africa
africa = acled_data[acled_data['region'].str.contains("Africa")]

## Conflicts by country
africa_conflicts = africa.groupby('country')['event_id_no_cnty'].count().reset_index(name='events_count').sort_values(['events_count'],ascending=False)

## Fatalities by country
africa_fatalities = africa.groupby('country')['fatalities'].sum().reset_index(name='fatalities_counts').sort_values(['fatalities_counts'],ascending=False)

## Event Types
africa_event_types = africa.groupby('event_type')['event_type'].count().reset_index(name='event_type_count').sort_values(['event_type_count'],ascending=False)

## Fatalities by month
fatalities_by_month = africa.groupby('year_mon')['fatalities'].sum().reset_index(name='fatalities_by_month')
fatalities_by_month['year_mon2'] = fatalities_by_month['year_mon'].astype(str)

africa_analytics_bars = st.container()
with africa_analytics_bars:
  bar_chart1, bar_chart2 = st.columns(2)

  top_ten_events = africa_conflicts[:10]
  fig1 = px.bar(top_ten_events, x='events_count', y='country',title="Conflict Events by Country",orientation='h')
  fig1.update_layout(yaxis={'categoryorder':'total ascending'}) 

  top_ten_fatalities = africa_fatalities[:10]
  fig2 = px.bar(top_ten_fatalities, x='fatalities_counts', y='country',title="Fatalities by Country",orientation='h')
  fig2.update_layout(yaxis={'categoryorder':'total ascending'}) 
  
  bar_chart1.plotly_chart(fig1)
  bar_chart2.plotly_chart(fig2)


africa_analytics_conflict_types = st.container()
with africa_analytics_conflict_types:
  bar_chart3,bar_chart4 = st.columns(2)

  fig3 = px.bar(africa_event_types, x='event_type_count', y='event_type',title="Distribution of Event Types",orientation='h')
  fig3.update_layout(yaxis={'categoryorder':'total ascending'}) 

  fig4 = px.line(fatalities_by_month, x="year_mon2", y="fatalities_by_month", title='Fatalities by Month')

  
  
  bar_chart3.plotly_chart(fig3)
  bar_chart4.plotly_chart(fig4)


continental_analysis_events = st.container()
with continental_analysis_events:

  Cumulative_cases_plot = px.choropleth(africa_conflicts,
                      locations="country", #Spatial coordinates and corrseponds to a column in dataframe
                      color="events_count", #Corresponding data in the dataframe
                      locationmode = 'country names', #location mode == One of ‘ISO-3’, ‘USA-states’, or ‘country names’ 
                      #locationmode == should match the type of data entries in "locations"
                      scope="africa", #limits the scope of the map to Africa
                      title ="Conflict Events Distribution in Africa",
                      hover_name="country",
                      color_continuous_scale = "deep",
                    )
  Cumulative_cases_plot.update_traces(marker_line_color="black") # line markers between states
  st.plotly_chart(Cumulative_cases_plot)


continental_analysis_fatalities = st.container()
with continental_analysis_fatalities:
  Cumulative_fatalities_plot = px.choropleth(africa_fatalities,
                      locations="country", #Spatial coordinates and corrseponds to a column in dataframe
                      color="fatalities_counts", #Corresponding data in the dataframe
                      locationmode = 'country names', #location mode == One of ‘ISO-3’, ‘USA-states’, or ‘country names’ 
                      #locationmode == should match the type of data entries in "locations"
                      scope="africa", #limits the scope of the map to Africa
                      title ="Conflict Fatality Distribution in Africa",
                      hover_name="country",
                      color_continuous_scale = "reds",
                    )
  Cumulative_fatalities_plot.update_traces(marker_line_color="black") # line markers between states
  st.plotly_chart(Cumulative_fatalities_plot)


st.subheader("Country Network Analysis")
st.write('To run network analysis on a particular country click the button below.')
if st.button('Run Network Analysis'):
  with st.spinner("Running Network Analysis"):
    
    
    st.write('Select country of interest for network analysis using knowledge graph')

    selected_country = st.selectbox(
        'Select Country for Network Analysis',
        africa['country'].unique())
    
    ## Filter for country
    africa_select = africa[['event_date','year','event_type','actor1','actor2','region','country','notes','fatalities']]

    country_data = africa_select[africa_select['country']==selected_country]

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

  st.success('Network analysis completed.')



