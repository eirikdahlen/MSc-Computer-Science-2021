print("\nRunning imports...")
import os
import re
import tweepy as tw
import pandas as pd 
from dotenv import load_dotenv
from datetime import date
from dateutil.relativedelta import relativedelta
import Preprocessing.preprocessing as mc
from Tags.get_tags import get_tags_to_search

class TwitterScraper:
    
    def __init__(self, collect_tweets_from_date='2021-02-17', monthly_time_span_for_tweets=6):
        self.api = self.setup_auth_to_api()
        self.since_date = collect_tweets_from_date
        self.filter_date = self.get_date_x_month_from(datestring=self.since_date, x=monthly_time_span_for_tweets)
        self.collected_users = set()

    def setup_auth_to_api(self):
        load_dotenv()

        CONSUMER_KEY = os.getenv('CONSUMER_KEY_TWITTER')
        CONSUMER_SECRET = os.getenv('CONSUMER_SECRET_TWITTER')
        ACCESS_TOKEN = os.getenv('ACCESS_TOKEN_TWITTER')
        ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET_TWITTER')

        auth = tw.OAuthHandler(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        api = tw.API(auth, wait_on_rate_limit=True)
        print('Authentication succeeded!\n---------------------------------\n')
        return api

    def create_twitter_dataset(self, user_ids, include_retweets=True, limit=10):
        """
        Traverse through users from input-tweets and the user they have retweeted, and create a list of lists dataset.
        
        Args:
            tweets ([Object]): [Tweepy ItemIterator]

        Returns:
            [List of lists]: [Datapoint in a list of lists manner]
        """
        data = []
        count = 0
        for user_id in user_ids:
            count += 1
            print(count, '/', len(user_ids), 'Working with:', user_id)
            user_tweets, users_from_retweets = self.get_tweets_from_user(user_id, limit=limit)
            
            if users_from_retweets:
                for user_id_from_rt in users_from_retweets:
                    retweeted_user_tweets, _ = self.get_tweets_from_user(user_id_from_rt, include_retweets=False, limit=limit)
                    user_tweets += retweeted_user_tweets
            
            data += user_tweets
        return data
        
        
    def get_user_id_from_search(self, search_words='', include_retweets=True, limit=100):
        """
        Examples:
        since_date: format YYYY-MM-DD
        search_words: "climate+change" will return all tweets that contain both of those words (in a row) in each tweet
        """
        if not include_retweets:
            search_words = search_words + " -filter:retweets"
        
        tweets = tw.Cursor(self.api.search,
                           q=search_words,
                           lang='en',
                           since=self.since_date,
                           tweet_mode='extended',
                           ).items(limit)
        user_ids = set(tweet.user.id_str for tweet in tweets)
        self.collected_users.update(user_ids)
        return user_ids

    def get_tweets_from_user(self, user_id, limit, include_retweets=True):
        try:
            tweets = tw.Cursor(self.api.user_timeline,
                            id=user_id,
                            count=limit,
                            lang='en',
                            include_rts=include_retweets,
                            tweet_mode='extended'
                            ).items(limit=limit)
        
            user_tweets = []
            users_from_retweets = []
            
            for tweet in tweets:
                if str(tweet.created_at).split()[0] > self.filter_date:
                    if tweet.full_text[0:2] == 'RT':
                        if include_retweets:
                            user_id_from_RT = self.get_id_from_screen_name(self.get_screen_name_from_retweet(tweet.full_text))
                            if user_id_from_RT and user_id_from_RT not in self.collected_users:
                                self.collected_users.add(user_id_from_RT)
                                users_from_retweets.append(user_id_from_RT)
                                print("Added user:", user_id_from_RT)
                        continue
                    
                    data = [str(tweet.id_str),
                            str(tweet.user.id_str),
                            str(tweet.user.name),
                            str(tweet.user.screen_name),
                            str(tweet.user.description),
                            str(tweet.full_text)]
                    
                    data = self.preprocess_tweet(data)
                    user_tweets.append(data)
        except tw.TweepError:
            print("Error...")
            return [], []
        return user_tweets, users_from_retweets
    
    def preprocess_tweet(self, data):
        data = mc.remove_whitespace(data)
        data = mc.to_lowercase(data)
        data = mc.handle_urls(data)
        data = mc.handle_mentions(data)
        data = mc.handle_emoji(data)
        data = mc.handle_character_entity_references(data)
        data = mc.handle_punctuation_and_numbers(data)
        data = mc.handle_abbreviations(data)
        data = mc.remove_non_ascii(data)
        return data

    def get_date_x_month_from(self, datestring, x=6):
        year, month, day = datestring.split('-')
        d = date(int(year), int(month), int(day)) + relativedelta(months=-x)
        return str(d)

    def get_screen_name_from_retweet(self, tweet):
        match = re.search('RT @(.*?):', tweet)
        if not match: return ''
        return match.group(1)
    
    def get_id_from_screen_name(self, screen_name):
        return self.api.get_user(screen_name).id_str
        
    def get_user_info(self, tweets):
        return [[tweet.user.id_str, tweet.user.name, tweet.user.screen_name, tweet.user.description, tweet.user.location, tweet.full_text] for tweet in tweets]
        
    def to_pandas_df(self, data, columns):
        return pd.DataFrame(data=data, columns=columns)
        
    def df_to_csv(self, dataframe, filename):
        dataframe.to_csv(filename)
        
    def save_user_ids_to_file(self, user_ids, filename):
        with open(filename, 'w') as file:
            for user_id in user_ids:
                file.write(user_id + '\n')
    
def main():
    names_from_file = True
    print("Initializing scraper with Auth...")
    scraper = TwitterScraper(collect_tweets_from_date='2021-03-04', monthly_time_span_for_tweets=6)
    
    
    user_ids = set()
    if names_from_file:
        with open('prorec_users.txt', 'r') as file:
            for line in file.readlines():
                user_ids.add(line.strip())
    else: 
        tags = get_tags_to_search(filename='Tags/pro_rec.txt')
        for tag in tags:
            print(tag)
            user_ids.update(scraper.get_user_id_from_search(search_words=tag,
                                                                include_retweets=True,
                                                                limit=100))
        scraper.save_user_ids_to_file(user_ids, 'prorec_users.txt')
    
    data = scraper.create_twitter_dataset(user_ids=user_ids, limit=50)
    
    df = scraper.to_pandas_df(data=data, columns=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text'])
    scraper.df_to_csv(dataframe=df, filename="twitter_data_prorec.csv") 
    
 
if __name__ == "__main__":
    print("Running main...")
    main()