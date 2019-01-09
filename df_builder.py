
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from scraping_code import *

def get_data(job, skill_set, city = None, state = None):

    '''
    Input: job (string), skill_set (list), city (string), state (string)
    Ouput:
        df: DataFrame including job info (DataFrame)
        unreadable_count: job postings that could not be opened and read
    '''

    job_urls = get_job_urls(job)

    if job_urls == 'Invalid Search':
        return 'Invalid Search'

    job = []
    unreadable_count = 0

    for url in job_urls:
        job_info = get_job_info(url)

        if job_info == "url could not be opened and read":
            unreadable_count += 1
        else:
            job_title, company_name, city, state, job_description = job_info
            text_no_skills, skills = clean_job_description(job_description,skill_set)
            job.append({'job_title':job_title,"company":company_name,'city':city, 'state': state,'desc': text_no_skills, 'skills':skills,'url': url})

    return pd.DataFrame(job), unreadable_count

def dfs_to_csv(job,file_name,skill_set,path):
    '''
    Input: job as typed out in indeed
    '''
    job_df, unreadable_count = get_data(job,skill_set)
    job_df['job'] = job
    job_df.to_csv(path + file_name + '.csv')

    return unreadable_count

#finish these
ds_count = dfs_to_csv('Data Scientist','data_scientist',skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
mle_count = dfs_to_csv('Machine Learning Engineer', 'machine_learning_engineer',skill_set, r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/' )
soft_developer_count = dfs_to_csv('Software Developer', 'software_developer',skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
soft_engineer_count = dfs_to_csv('Software Engineer', 'software_engineer',skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
fsd_count = dfs_to_csv('Full Stack Developer', 'full_stack_developer',skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
bia_count = dfs_to_csv('Business Intelligence Analyst','business_intelligence_analyst', skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
data_eng_count = dfs_to_csv("Data Engineer", 'data_engineer', skill_set, r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
data_arc_count = dfs_to_csv('Data Architect', 'data_architect',skill_set,r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
dev_ops_count = dfs_to_csv('DevOps','dev_ops',skill_set, r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
python_count = dfs_to_csv('Python','python',skill_set, r'/Users/danielatejada 1/Desktop/Galvanize/capstone_project/scraping_data/')
