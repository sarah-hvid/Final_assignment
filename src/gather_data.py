"""
A script for scraping and preprocessing the Pokemon generation data from wikipedia.
"""

# tools for scraping
import requests 
from bs4 import BeautifulSoup

# data tools
import pandas as pd
import os
import numpy as np
import pandas as pd
import re


# function to scrape the wikipedia site
def scrape_wiki():
    # get the response in the form of html
    wikiurl = "https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon"
    table_class = "wikitable sortable jquery-tablesorter"
    response = requests.get(wikiurl)
    
    return response


# function to convert the html table to a dataframe
def html_to_df(response):
    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    gen_table = soup.find('table',{"style": "width:100%; border:2px solid grey"})
    
    # converting the list to a dataframe
    df = pd.read_html(str(gen_table))
    df = pd.DataFrame(df[0])
    
    # cleaning the dataframe and saving it
    df.columns = ['_'.join(col) for col in df.columns]
    df.to_csv('data/pokemon_wikitable.csv', index = False)
    
    return


# function to fix the names of the Pokemon in 
def fix_strings(df):
    # removing most of the special characters and whitespace
    df['Name'] = df['Name'].str.lower()
    df['Name'] = df['Name'].str.replace(r'\[\w\]', '', regex=True)
    df['Name'] = df['Name'].str.replace('♀', '-f', regex=True)
    df['Name'] = df['Name'].str.replace('♂', '-m', regex=True)
    df['Name'] = df['Name'].str.replace(r'[^-éa-zA-Z0-9\s]', '', regex=True)
    df['Name'] = df['Name'].str.replace(r' +', '', regex=True)
    df['Name'] = df['Name'].str.replace(r'é', 'e', regex=True)

    # correcting specific names
    df['Name'] = df['Name'].str.replace('mrmime', 'mr-mime', regex=True)
    df['Name'] = df['Name'].str.replace('mimejr', 'mime-jr', regex=True)
    df['Name'] = df['Name'].str.replace('typenull', 'type-null', regex=True)
    df['Name'] = df['Name'].str.replace('tapukoko', 'tapu-koko', regex=True)
    df['Name'] = df['Name'].str.replace('tapulele', 'tapu-lele', regex=True)
    df['Name'] = df['Name'].str.replace('tapubulu', 'tapu-bulu', regex=True)
    df['Name'] = df['Name'].str.replace('tapufini', 'tapu-fini', regex=True)
    
    return df


# function to wrangle the wiki dataframe
def clean_wiki_dataframe(df):
    # clean column names
    df = df[['Generation I_Name', 'Generation II_Name', 'Generation III_Name', 'Generation IV_Name', 'Generation V_Name', 'Generation VI_Name', 'Generation VII_Name']]
    df.columns = ['1', '2', '3', '4', '5', '6', '7']
    
    # pivot dataframe to long format
    df = df.reset_index()
    df = pd.melt(df, value_vars=['1', '2', '3', '4', '5', '6', '7'])
    df.columns = ['Generation', 'Name']
    
    # correcting names and dropping na's for merge
    df = fix_strings(df)
    
    df.loc[df.Name == 'victini', 'Generation'] = 5
    
    df = df.loc[df['Name'] != 'noadditionalpokemon']
    df = df[df['Name'].notna()]
    
    return df


# function to clean the kaggle dataframe
def clean_dataframe(df1):
    # clean the first dataframe
    df1['Name'] = df1['Name'].str.replace('-[pasrci5]\w*', '', regex=True)
    df1['Name'] = df1['Name'].str.replace('-no\w*|-ma\w*|-la\w*|-bl\w*|-ord\w*|-ba\w*|-mid\w*|-met\w*', '', regex=True)
    
    return df1


def main():
    # scrape the data and fix the html format
    response = scrape_wiki()
    html_to_df(response)

    # read in the dataframes
    filepath = os.path.join('data/pokemon_wikitable.csv')
    df = pd.read_csv(filepath)

    filepath = os.path.join('data/pokemon.csv')
    df1 = pd.read_csv(filepath)
    
     #clean dataframes
    df = clean_wiki_dataframe(df)
    df1 = clean_dataframe(df1)
    
    # create the final dataframe and save it
    df2 = pd.merge(df1, df, on='Name')
    df2 = df2.sort_values(by=['Name'])
    df2.to_csv('data/pokemon_all.csv', index = False)
    
    print('Data saved.')
    
    return


if __name__ == '__main__':
    main()