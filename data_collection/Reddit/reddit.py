import os
import praw
import pandas as pd
from dotenv import load_dotenv
import Preprocessing.preprocessing as mc
from langdetect import detect



def setup_auth_to_api():
    load_dotenv()

    CLIENT_ID = os.getenv('CLIENT_ID_REDDIT')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET_REDDIT')
    USER_AGENT = os.getenv('USER_AGENT_REDDIT')

    api = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    return api


def get_subreddit_hot_posts(api, subreddit, limit=10):
    return api.subreddit(subreddit).hot(limit=limit)

def search_keyword(api, search_word):
    posts = []
    for submission in api.subreddit("all").search(search_word, limit=100):
        posts.append(submission)
    return posts

def get_user_info(posts):
    return [[post.id, post.title, post.subreddit, post.selftext.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')] for post in posts]

def to_pandas_df(data, columns):
    return pd.DataFrame(data=data, columns=columns)

def df_to_csv(dataframe, filename):
    dataframe.to_csv(filename)

def main():
    keywords = ['pro-ED', 'proana', 'promia', 'thinspo', 'bonespo', 'hipbones', 'ugw']
    api = setup_auth_to_api()
    posts = []
    for keyword in keywords:
        posts = [*posts, *search_keyword(api=api, search_word=keyword)]
    user_info = get_user_info(posts)
    df = to_pandas_df(data=user_info, columns=['post_id', 'title', 'subreddit', 'text'])    
    
    # Preproccesing
    processed = []
    for post in user_info:
        if len(post[3].split()) < 3:
            continue
        print('\n')
        print('New post!!! ')
        print('\n')
        print(post[3])
        if detect(post[3]) != 'en':
            continue
        p = preprocess_text(['', '', '', '', post[1], post[3]])
        processed.append([post[0], post[2], p[4] + ' // ' + p[5]])
    df_processed = to_pandas_df(data=processed, columns=['post_id', 'subreddit', 'text'])
    
    df.to_csv('reddit_data.csv')
    df_processed.to_csv('reddit_processed.csv')

def preprocess_text(data):
    data = mc.to_lowercase(data)
    data = mc.handle_urls(data)
    data = mc.handle_emoji(data)
    data = mc.handle_character_entity_references(data)
    data = mc.handle_punctuation_and_numbers(data)
    data = mc.remove_non_ascii(data)
    return data

if __name__ == "__main__":
    main()
