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
from scraping_code import get_job_info
from scraping_code import clean_job_description
from scraping_code import skill_set
from top_jobs import *

def get_ideal_text(skill_set,ideal_url):
    '''
    Input:
        skill set: skill_set (list)
        ideal_url: indeed url of ideal job posting (string)
    Output: job description without skill set (list of strings)
    '''

    _, _, _, _, job_description = get_job_info(ideal_url)
    text_no_skills, _ = clean_job_description(job_description,skill_set)

    return text_no_skills

def create_df():
    '''
    Input: None
    Output: list of Dataframes (list)
    '''
    df_ds = pd.read_csv('scraping_data/data_scientist.csv')
    df_mle = pd.read_csv('scraping_data/machine_learning_engineer.csv')
    df_sd = pd.read_csv('scraping_data/software_developer.csv')
    df_fsd = pd.read_csv('scraping_data/full_stack_developer.csv')
    df_bia = pd.read_csv('scraping_data/business_intelligence_analyst.csv')
    df_de = pd.read_csv('scraping_data/data_engineer.csv')
    df_da = pd.read_csv('scraping_data/data_architect.csv')
    df_py = pd.read_csv('scraping_data/python.csv')
    df_devops = pd.read_csv('scraping_data/dev_ops.csv')
    df_se = pd.read_csv('scraping_data/software_engineer.csv')

    dfList = [df_ds,df_mle,df_sd,df_fsd,df_bia,df_de,df_da,df_py,df_devops,df_se]

    return dfList

def top_jobs(ideal_url, top_num = 10, job = None, city = None, state = None, skills = None, stringency = None):
    '''
    Input:
        ideal_url: indeed ideal job url (string)
        top_num: number of jobs to be displayed (int)
        job: job name for indeed query (string)
        city: city of desired job (string)
        state: state of desired job (string)
        skills: skill set possessed (list of strings)
        stringency: how to compare job skill set and personal skill set (string)
    Output:
        top_df: DataFrame featuring top_num most similar jobs (DataFrame)
    '''
    ideal_text = get_ideal_text(skill_set,ideal_url)
    dfList = create_df()
    joint_df = join_dfs(dfList)
    formatted_df = format_df(joint_df)
    query_df = get_query_df(formatted_df,job,city,state,skills,stringency)

    if type(query_df) == str:
        return query_df
    else:
        cosine_sim, indices = get_top_sim(ideal_text, query_df,top_num)
        top_df = get_top_jobs(query_df, indices, cosine_sim)
        top_df = top_df.reset_index(drop=True)
        return top_df
