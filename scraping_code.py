# CHECK WHICH OF THESE ARE ACTUALLY BEING USED
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def url_split_join(arg):
    '''
    Input: job
    Output: text formatted for indeed search
    '''
    arg = arg.split()
    return '+'.join(word for word in arg)

def get_search_url(job,city = None, state= None):

    '''
    Input: job and city
    Output: url that directs to results page for the query
    '''

    job = url_split_join(job)
    site_list = ['http://www.indeed.com/jobs?q="', job, '"']

    return ''.join(site_list)

def get_job_urls(job,city = None, state = None):

    '''
    Goes through the result page for the query and return for urls for each organic job posting
    Input: query
    Output: list of all jobs urls features in the query
    '''

    search_url = get_search_url(job,city, state) #gets results page

    try:
        site = urlopen(search_url).read()
    except:
        return 'Invalid Search' #raises exception if search combination is invalid of if no jobs of that nature exist

    soup = BeautifulSoup(site)

    if len(soup) == 0: # in case the default parser lxml doesn't work, try another one
        soup = BeautifulSoup(site, 'html5lib')


   #gets the total number (organic and sponsored) of job postings
    num_jobs = soup.find(id = 'searchCount').string
    num_jobs = re.findall('\d+', num_jobs)
    num_jobs = int("".join(num_jobs[1:]))

    #gets the number of page results
    if num_jobs > 10:
        num_pages = num_jobs//10
    else:
        num_pages = 1

    page_urls = []

    #iterates over each page to get the urls within that page
    for i in range(num_pages):

        start_num = str(i*10)  #page 1 starts at start = 0 , page 1 starts at 10 etc.
        page_url = ''.join([search_url,'&start=', start_num])

        current_page = urlopen(page_url).read()
        page_soup = BeautifulSoup(current_page)

        if len(page_soup) == 0: # In case the default parser lxml doesn't work, try another one
            page_soup = BeautifulSoup(page_url, 'html5lib')

        results_col =  page_soup.find(id = 'resultsCol')
        organic_tags = results_col.find_all('div', {'data-tn-component' : "organicJob"}) #gets tags for organic rearch results

        urls  = [x.a.attrs.get('href') for x in organic_tags] #gets the url for the specific job
        page_urls.append(urls)

        if len(urls) < 10: #necessary because sponsored jobs results included in num_jobs
            break

    job_urls = ['https://www.indeed.com'+job for sublist in page_urls for job in sublist]

    return job_urls

def get_job_info(job_url):

    '''
    Input: url of indeed job posting
    Output: role, title, location and list of words in description
    '''

    #TO DO: 1) check that ds3 works

    try:
        site =  urlopen(job_url).read() #opens and returns html
    except:
        return "url could not be opened and read" #CHECK THIS

    soup = BeautifulSoup(site)

    if len(soup) == 0: # In case the default parser lxml doesn't work, try another one
        soup = BeautifulSoup(site, 'html5lib')

    #general job information

    try:
        job_title = soup.find('h3',{'class':"icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"}).get_text()
    except:
        job_title = 0

    try:
        company_name = soup.find('div',{'class':'icl-u-lg-mr--sm icl-u-xs-mr--xs'}).get_text() #do the other company name thing, do this for state
    except:
        company_name = None

    try:
        company_info = soup.select('div.jobsearch-InlineCompanyRating.icl-u-xs-mt--xs.jobsearch-DesktopStickyContainer-companyrating')[0].text
        company_info = company_info.split('-')[-1]
        company_info = company_info.split(' ')

        info = []

        for i in company_info:
            try:
                i = int(i)
                info.append(i)
            except:
                info.append(i)

        location = [x for x in info if not isinstance(x, int)]
        city = ' '.join(location[:-1])
        state = location[-1]

    except:
        city = 'unavailable'
        state = 'unavailable'

    try:
        content = soup.find('div',{'class':'jobsearch-JobComponent-description icl-u-xs-mt--md'})
        words = content.get_text().split()

        punctuation = string.punctuation
        stop_words = stopwords.words('english')

        words =[''.join(ch for ch in word if ch not in punctuation) for word in words] #gets rid of punctuation between words to enable joint word adjustment and genereal punctuation
        words = [re.sub(r"([a-z])([A-Z])", r"\1 \2",word).split() for word in words] #adjusts for joint words

        words =[word.lower() for sublist in words for word in sublist] #flattens lists
        job_description = [word for word in words if word not in stop_words and word not in punctuation] #gets rids of stop words

    except:
        job_title = None
        company_name = None
        state = None
        job_description = None

    return job_title, company_name, city[:-1], state, job_description

def clean_job_description(job_description,skill_set):
    '''
    Input: list of words included in job posting
    Output:
        Desc = list of stemmed words included in job posting no including tecnical skills
        skills = technical skills required for job

    '''
    #TO DO      Check whether the skills should be taken out
    #           The description includes no skills because of the variety of languages can be used to do the same job
    #           Consider whether you want to add back the skills. Because maybe skills are very reflective of nature of job
    #           GET RID OF STEMMER AND PUT IT IN THE SIMILARITY FILE
    if job_description not None:
        skills = list(set([word for word in job_description if word in skill_set]))
        text_no_skills = [word for word in job_description if word not in skills]

        #stemmer = SnowballStemmer('english') #get rid if this if you're going to do semantic similary
        #text_no_skills = [stemmer.stem(word) for word in text_no_skills]
    else:
        skills = None
        text_no_skills = None

    return text_no_skills, skills

def get_data(job, skill_set, city = None, state = None):
    '''
    Input:
    Ouput: dataframe with revelant information about job
        Words used for similiarty
        Skills and location used for filetring results
    '''

     #TO DO 1) how do i make this run faster

    job_urls = get_job_urls(job,city)

    if job_urls == 'Invalid Search':
        return 'Invalid Search'

    job = []
    unreadable_count = 0

    for url in job_urls:
        print(url)
        job_info = get_job_info(url)

        if job_info == "url could not be opened and read":
            unreadable_count += 1
        else:
            job_title, company_name, city, state, job_description = job_info
            text_no_skills, skills = clean_job_description(job_description,skill_set)
            job.append({'job_title':job_title,"company":company_name,'city':city, 'state': state,'desc': text_no_skills, 'skills':skills,'url': url})

    return pd.DataFrame(job), unreadable_count
