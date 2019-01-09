from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

skill_set = ['r','python','java','c++','ruby','perl','matlab','javascript','scala','excel','tableau','d3js','sas','spss','d3','hadoop',
            'mapreduce','spark','pig','hive','shark','zookeeper','flume','mahout','sql','nosql','hase','cassandra','mongodb','docker','aws']

def url_split_join(job):

    '''
    Input: job query (string)
    Output: text formatted for indeed search (string)
    '''

    job_format = job.split()
    return '+'.join(word for word in job_format)

def get_search_url(job):

    '''
    Input: job query (string)
    Output: indeed url for job query (string)
    '''

    job = url_split_join(job)
    site_list = ['http://www.indeed.com/jobs?q="', job, '"']

    return ''.join(site_list)

def get_job_urls(job):

    '''
    Input: job name (query)
    Output: list of organic (not sponsored) indeed job postings (urls) for the given job query (list of strings)
    '''

    search_url = get_search_url(job) #gets results page

    try:
        site = urlopen(search_url).read()
    except:
        return 'Invalid Search' #raises exception if search combination is invalid of if no jobs of that nature exist

    soup = BeautifulSoup(site, features="lxml")

    if len(soup) == 0: # in case the default parser lxml doesn't work, try another one
        soup = BeautifulSoup(site, 'html5lib')

    #gets the total number of job postings (organic and sponsored)

    try:
        num_jobs = soup.find(id = 'searchCount').string
        num_jobs = re.findall('\d+', num_jobs)
        num_jobs = int("".join(num_jobs[1:]))

    except:
        num_jobs = 150000

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
        page_soup = BeautifulSoup(current_page, features="lxml")

        if len(page_soup) == 0: # In case the default parser lxml doesn't work, try another one
            page_soup = BeautifulSoup(page_url, 'html5lib')

        try:

            results_col =  page_soup.find(id = 'resultsCol')
            organic_tags = results_col.find_all('div', {'data-tn-component' : "organicJob"}) #gets tags for organic rearch results

            urls  = [x.a.attrs.get('href') for x in organic_tags] #gets the url for the specific job
            page_urls.append(urls)

            if len(urls) < 10: #necessary because sponsored jobs results included in num_jobs
                break
        except:
            break

    job_urls = ['https://www.indeed.com'+job for sublist in page_urls for job in sublist]

    return job_urls

def get_job_info(job_url):

    '''
    Input: indeed job posting url (string)
    Output: job_title (string), company_name (string), city (string), state (string) and job description (list of words) for job posting
    '''

    try:
        site =  urlopen(job_url).read() #opens and returns html
    except:
        return "url could not be opened and read"

    soup = BeautifulSoup(site, features="lxml")

    if len(soup) == 0: # in case the default parser lxml doesn't work, try another one
        soup = BeautifulSoup(site, 'html5lib')

    #general job information

    try:
        job_title = soup.find('h3',{'class':"icl-u-xs-mb--xs icl-u-xs-mt--none jobsearch-JobInfoHeader-title"}).get_text()
    except:
        job_title = 'unavailable'

    try:
        company_name = soup.find('div',{'class':'icl-u-lg-mr--sm icl-u-xs-mr--xs'}).get_text() #do the other company name thing, do this for state
    except:
        company_name = 'unavailable'

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
        job_description = 'Unavailable'

    return job_title, company_name, city[:-1], state, job_description

def clean_job_description(job_description,skill_set):

    '''
    Input: job description (list of Words), skill_set (list)
    Output:
        text_no_skills: stemmed job description excluding technical skills included in skill set (list of words)
        skills: technical skills included in skill set (list of words)

    '''

    try:
        skills = list(set([word for word in job_description if word in skill_set]))
        text_no_skills = [word for word in job_description if word not in skills]

    except:
        skills = 'unavailable'
        text_no_skills = 'unavailable'

    return text_no_skills, skills
