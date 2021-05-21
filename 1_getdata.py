# project3_1_getdata.py
#
# first written and tested in project3_1_getdata.ipynb
# 
# creates ./data/redditcomments.csv, saved after each pull of 100 comments.
#

# import the basics, + request:
import requests
import pandas as pd
import numpy as np
from time import sleep

# basic url for my data. I am using comments because I am interested in the conversations:
url0 ='https://api.pushshift.io/reddit/search/comment'

# subreddits:
# r/fantasy, 2.3m members, created 2008
# r/scifi, 1.3m membrs, created 2008
subreddits = ['fantasy', 'scifi']

# initialize my lists:
comments = [] # text of comments
dates    = [] # UTC created value for each comment
types    = [] # subreddit value: "fantasy" or "scifi"

# loop over my 2 subreddits
for subreddit in subreddits:
    
    #parameter dictionary of which subreddit to pull, and set to 100 posts (seems to be the max):
    params = {
        'subreddit': subreddit,
        'size': 100
    }

    # make 100 separate pulls. 100 pulls x 100 comments/pull = 10_000 comments per subreddit:
    for n in range(100):

        # get the data:
        r = requests.get(url0, params)

        # make lists for the comment text, UTC created date, and subreddit name for each new pull:
        new_body=[i['body'] for i in r.json()['data']]
        new_date=[i['created_utc'] for i in r.json()['data']]
        new_type=[subreddit for i in r.json()['data']]
        
        # add these new lists to the dataset:
        comments += new_body
        dates    += new_date
        types    += new_type
        
        # set the "before" parameter so the next pull is earlier comments:
        params['before'] = np.min(new_date)

        # print out some output so I know it's working:
        num = len(comments)
        print(f'Subreddit : {subreddit}, retrieved {num} comments total.')
        
        # make my dataframe:
        d = {}
        d['comment']=comments
        d['date']   =dates
        d['types']  =types
        df = pd.DataFrame(d)
        
        # save to disk. I'm doing this after every pull in case I run into errors:
        df.to_csv('./data/redditcomments.csv')

        # Pause between pulls so I don't set off any defenses:
        sleep(5)
