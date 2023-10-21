import pandas as pd
import dateutil.parser
import warnings

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer

from util import extract_time, check_degree_type, clean_text
from constants import DEGREE_LIST, CURRENT_DATE, TITLE_KEYWORDS


def extract_row_features(row):
    features = {}
    features.update(extract_years_experience(row['experience']))
    features.update(extract_job_titles(row['experience']))     
    features.update(extract_degree_types(row['education']))
    return features


def extract_years_experience(experience_list):
    feature = {}
    if isinstance(experience_list, (list, tuple)):
        # Extract (start_time, date_time) tuples from experience_list
        time_list = [extract_time(experience['time']) for experience in experience_list if isinstance(experience, dict) and 'time' in experience]
        # Filter out tuples where start_time is None
        time_list = [time for time in time_list if time[0] is not None]
        # Sort by start_time
        time_list = sorted(time_list, key=lambda x: x[0])
        # Compute non-overlapping years of experience
        curr_end_time = dateutil.parser.parse('1900-01-01')
        years_experience = 0
        for start_time, end_time in time_list:
            effective_start_time = max(start_time, curr_end_time)
            years_experience += max(0, (end_time - effective_start_time).days / 365)
            curr_end_time = max(curr_end_time, end_time)

        feature['years_experience'] = years_experience
        feature['num_jobs'] = len(experience_list)
    else:
        warnings.warn('experience_list is not a list or tuple')
    return feature


def extract_degree_types(education_list):
    degree_dict = {degree_type: 0 for degree_type in DEGREE_LIST}
    if isinstance(education_list, (list, tuple)):
        for education in education_list:
            if isinstance(education, dict) and 'degree' in education:
                degree_type = check_degree_type(education['degree'])
                degree_dict[degree_type] = 1
    else:
        warnings.warn('education_list is not a list or tuple')
    return degree_dict


def extract_job_titles(experience_list):
    '''
    Extract job titles from experience_list that are within 5 years of CURRENT_DATE
    '''
    if isinstance(experience_list, (list, tuple)):
        features = {}
        # Extract the max end_time from experience_list
        time_periods = [extract_time(experience['time']) if isinstance(experience, dict) and 'time' in experience else (None, None) for experience in experience_list]
        all_job_titles = [experience['title'] if isinstance(experience, dict) and 'title' in experience else "" for experience in experience_list]
        # Get the max_end_time, ignoring None values
        max_end_time = max([time_period[1] for time_period in time_periods if time_period[1] is not None], default=None)

        recent_job_titles = []

        for i, time_period, job_title in zip(range(len(time_periods)), time_periods, all_job_titles):
            _, end_time = time_period
            if end_time is not None and end_time >= max_end_time:
                # Include the most recent jobs
                recent_job_titles.append(job_title)
            elif end_time is None and i == 0:
                # Recent job titles are listed first
                recent_job_titles.append(job_title)
            
        # Convert job_titles to a string
        recent_job_titles = ' '.join(recent_job_titles)
        # Clean the text
        recent_job_titles = clean_text(recent_job_titles)
        # Remove stopwords
        recent_job_titles = ' '.join([word for word in recent_job_titles.split() if word not in stopwords])

        features['job_titles'] = recent_job_titles
    else:
        features['job_titles'] = ''
        warnings.warn('experience_list is not a list or tuple')
    return features