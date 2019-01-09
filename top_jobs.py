import pandas as pd
import ast
from functools import reduce
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import string
from urllib.request import urlopen
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re


def join_dfs(dfList):

    '''
    Input: list of DataFrames to be joined (list of DataFrames)
    Output: merged DataFrame (DataFrame)
    '''

    return reduce(lambda x, y: x.append(y), dfList)

def format_df(df):

    '''
    Input: DataFrame
    Output: formatted DataFrame
    '''

    #dropping unnamed
    df = df.iloc[:,1:]
    df = df.reset_index(drop=True)

    #dropping duplicated rows
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    #getting rid of jobs that have no descriptions
    index = df[df['desc'] == "['U', 'n', 'a', 'v', 'a', 'i', 'l', 'a', 'b', 'l', 'e']"].index
    df = df.drop(index,axis = 0)
    df = df.reset_index(drop=True)

    #nan cities are remote
    df['city'] = df.city.fillna('Multiple')

    #formatting for alternate locations
    df['city'] = df.city.apply(lambda x: 'Multiple' if x == "Unite" else x)
    df['city'] = df.city.apply(lambda x: 'unavailable' if x == "unavailabl" else x)
    df['city'] = df.city.apply(lambda x: 'unavailable' if x == "unavailabl" else x)
    df['city'] = df.city.apply(lambda x: 'Chapel Hill' if x == "Nort" else x)
    df['city'] = df.city.apply(lambda x: 'Remote' if x == "Hom" else x)
    df['city'] = df.city.apply(lambda x: 'Remote' if x == "Home" else x)
    df['city'] = df.city.apply(lambda x: 'New York' if x == "New Yor" else x)
    df['city'] = df.city.apply(lambda x: 'San Juan' if x == "Puert" else x)
    df['city'] = df.city.apply(lambda x: 'Multiple' if x == "Rhod" else x)


    df['state'] = df.state.apply(lambda x: 'Multiple' if x == "States" else x)
    df['state'] = df.state.apply(lambda x: 'NY' if x == "State" else x)
    df['state'] = df.state.apply(lambda x: 'Remote' if x == "Home" else x)
    df['state'] = df.state.apply(lambda x: 'Remote' if x == "Based" else x)
    df['state'] = df.state.apply(lambda x: 'MN' if x == 'Minnesota' else x)
    df['state'] = df.state.apply(lambda x: 'AZ' if x == 'Arizona' else x)
    df['state'] = df.state.apply(lambda x: 'PA' if x == 'Pennsylvania' else x)
    df['state'] = df.state.apply(lambda x: 'FL' if x == 'Florida' else x)
    df['state'] = df.state.apply(lambda x: 'CA' if x == 'California' else x)
    df['state'] = df.state.apply(lambda x: 'TN' if x == 'Tennessee' else x)
    df['state'] = df.state.apply(lambda x: 'AR' if x == 'Arkansas' else x)
    df['state'] = df.state.apply(lambda x: 'TX' if x == 'Texas' else x)
    df['state'] = df.state.apply(lambda x: 'CT' if x == 'Connecticut' else x)
    df['state'] = df.state.apply(lambda x: 'NC' if x == 'Carolina' else x)
    df['state'] = df.state.apply(lambda x: 'AL' if x == 'Alabama' else x)
    df['state'] = df.state.apply(lambda x: 'MA' if x == 'Massachusetts' else x)
    df['state'] = df.state.apply(lambda x: 'GA' if x == 'Georgia' else x)
    df['state'] = df.state.apply(lambda x: 'IL' if x == 'Illinois' else x)
    df['state'] = df.state.apply(lambda x: 'VA' if x == 'Virginia' else x)
    df['state'] = df.state.apply(lambda x: 'PR' if x == 'Rico' else x)
    df['state'] = df.state.apply(lambda x: 'KS' if x == 'Kansas' else x)
    df['state'] = df.state.apply(lambda x: 'RI' if x == 'Island' else x)
    df['state'] = df.state.apply(lambda x: 'MS' if x == 'Mississippi' else x)


    #filling nan
    df.fillna('unavailable')

    return df

def get_query_df(df,job = None, city = None, state = None, skills = None, stringency = None):
    '''
    Input:
        df: DataFrame
        job: desired indeed job (string)
        city: city of desired job (string)
        state: state of desired job (string)
        skills: skill set desired (string)
        stringency: how to compare job skills and skills posses (string)
    '''

    if job != None:

        jobs_available = df.job.unique()

        if job not in jobs_available:
            return 'Job Unavailable'

        df = df[df['job'] == 'Data Scientist']

    if city != None:
        cities_available = df.city.unique()

        if city not in cities_available:
            return 'City Unavailable'
        else:
            df = df[df['city'].isin([city,'Multiple'])]

    if state != None:

        states_available = df.state.unique()

        if state not in states_available:
            return 'State Unavailable'

        df = df[df['state'].isin([state,'Multiple'])]

    if skills != None:

        job_skills = df.skills.values
        job_skills = [ast.literal_eval(x) for x in job_skills]
        indexes = []

        if stringency == 'all':
            for i,job in enumerate(job_skills):
                if skills == job:
                    indexes.append(i)

            df = df.iloc[indexes,:]

        elif stringency == 'only':
            for i, job in enumerate(job_skills):
                if set(job).issubset(skills):
                    indexes.append(i)

            df = df.iloc[indexes,:]

        elif stringency == 'atleast':
            for i, job in enumerate(job_skills):
                if set(skills).issubset(job):
                    indexes.append(i)

            df = df.iloc[indexes,:]
        else:
            return 'Strigency must be: "all", "only" or "atleast"'

    if df.empty == True:
        return 'There are no jobs available for this combination'

    df = df.reset_index(drop=True)

    return df

def get_top_sim(ideal_text, query_df,num):
    '''
    Input:
        ideal_text: (list of strings)
        query_df: (DataFrames)
        num: (int)
    Output:
        cosine_sim: consine similarity of num number of jobs
        indices: indexes of num most similar jobs
    '''

    ideal_words = [" ".join(ideal_text)]

    job_words = query_df.desc.values
    job_words = [ast.literal_eval(x) for x in job_words] #turns string into a list
    job_words = [x for x in job_words if not isinstance(x, int)] #gets rid of numbers


    stemmer = SnowballStemmer('english') #get rid if this if you're going to do semantic similary
    job_words_stem = []
    for job in job_words:
        job_words_stem.append([stemmer.stem(word) for word in job])

    job_desc = [" ".join(li) for li in job_words_stem]

    vectorizer = TfidfVectorizer() #make sure these defaults are correct
    model = vectorizer.fit(job_desc)

    ideal_tfidf = model.transform(ideal_words)
    job_tfidf = model.transform(job_desc)

    cosine_sim = cosine_similarity(ideal_tfidf,job_tfidf)[0]

    if len(cosine_sim) <= num:
        return cosine_sim, cosine_sim.argsort()[::-1]
    else:
        return cosine_sim, cosine_sim.argsort()[::-1][0:num]

def get_top_jobs(query_df,indices, cosine_sim):
    '''
    Input:
        query_df: DataFrame
        indices: indices of most similar top num jobs
        cosine_sim: cosine similarity of top num job
    Output:
        top_df: DataFrame with information about the top num most similar jobs (DataFrame)
    '''

    top_jobs = query_df.iloc[indices,:]

    return top_jobs[['job_title','company','state','city','skills','url']]
