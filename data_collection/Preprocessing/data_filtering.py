import pandas as pd
import langid
import re


def is_english_post(post):
    try:
        language = langid.classify(post)
        if language[0] == 'en':
            return True
        return False
    except:
        print("Error in language detect...")
        return False


def is_short_post(post):
    try:
        post_edited = post.replace('MENTION', '').replace('URL', '')
        post_edited = re.sub(r"EMOJI(\w+)", '', post_edited)
        post_edited = post_edited.split()
        if len(post_edited) < 3:
            return True
        return False
    except:
        print("Error shortening post...")
        return False


def filter_short_posts(df, column_to_check):
    is_short = df[column_to_check].apply(lambda post: is_short_post(post))
    return df[~is_short]


def filter_non_english_post(df, column_to_check):
    is_english = df[column_to_check].apply(lambda post: is_english_post(post))
    return df[is_english]


def csv_to_df(filename, columns):
    return pd.read_csv(filename, names=columns)


def merge_datasets(newfile, filename1, filename2):
    with open(newfile, 'w') as newfile:
        with open(filename1, 'r') as file1:
            for line in file1.readlines():
                newfile.write(line)
        with open(filename2, 'r') as file2:
            for line in file2.readlines():
                newfile.write(line)


def shuffle_csv(filename, columns, new_filename):
    df = ''
    if len(columns) != 0:
        df = pd.read_csv(filename, names=columns)
    else: 
        df = pd.read_csv(filename)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(new_filename, index=False)


def sample_n_posts(toFile, fromFile, n, start_index=0):
    with open(toFile, 'a') as new_file:
        with open(fromFile, 'r') as prev_file:
            for line in prev_file.readlines()[start_index:start_index + n + 1]:
                new_file.write(line)


def split_dataset_for_annotation(fromFile, rows_in_file, n_overlapping, new_filename1, new_filename2):
    end_file1 = int(rows_in_file / 2 + n_overlapping / 2 + 1)
    start_file2 = int(rows_in_file / 2 - n_overlapping / 2)

    with open(fromFile, 'r') as file:
        header = file.readline()
        with open(new_filename1, 'w') as new_file1:
            new_file1.write(header)
            for line in file.readlines()[1:end_file1]:
                new_file1.write(line)
    with open(fromFile, 'r') as file:
        with open(new_filename2, 'w') as new_file2:
            new_file2.write(header)
            for line in file.readlines()[start_file2:]:
                new_file2.write(line)


def find_keywords_occurence(keywords, post):
    for word in keywords:
        if word in post:
            return True
    return False


def find_post_from_specific_users(df, keywords, drop_from):
    df = df.iloc[drop_from:]
    is_relevant = df['text'].apply(lambda post: find_keywords_occurence(keywords, post))
    new_df = df[is_relevant]
    users = set(new_df['user_id'].tolist())
    return df.loc[df['user_id'].isin(users)]


def get_post_ids(df, column):
    post_ids = df[column].tolist()
    return set(post_ids)


def check_id(post_id, post_ids):
    if post_id not in post_ids:
        return True
    return False


def extract_targeted_posts(df, keywords, post_ids):
    """
    seen_before = df['tweet_id'].apply(lambda tweet_id: check_id(tweet_id, post_ids))
    new_df = df[seen_before]
    
    tw_ids = set(new_df['tweet_id'].tolist())
    new_df = df.loc[df['tweet_id'].isin(tw_ids)] 
    """
    df = df.iloc[3001:]
    is_relevant = df['text'].apply(lambda post: find_keywords_occurence(keywords, post))
    relevant_df = df[is_relevant]
    ids = set(relevant_df['tweet_id'].tolist())
    return df.loc[df['tweet_id'].isin(ids)]


def main():
    #df = csv_to_df('twitter_data_prorec.csv', columns=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text'])
    #print(len(df.index))
    #df = filter_non_english_post(df, column_to_check='text')
    #print(len(df.index))
    #df = filter_short_posts(df, column_to_check='text')
    #print(len(df.index))
    #df.to_csv('twitter_data_prorec_filtered.csv')
    
    #df = csv_to_df('twitter_data_proed_filtered.csv', columns=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text'])
    #proed_df = find_post_from_specific_users(df=df, keywords=['proana', 'thinspo', 'meanspo', 'bonespo', 'edtwt'], drop_from=5000)
    #proed_df.to_csv('proed_tweets.csv')
    #sample_n_posts(toFile='prorec_tweets_sample_2000.csv', fromFile='twitter_data_prorec_filtered.csv', n=2000, start_index=5000)
    #sample_n_posts(toFile='twitter_prorec_samples.csv', fromFile='twitter_data_prorec_filtered.csv', n=5000)
    #merge_datasets(newfile='twitter_data_5000_sample.csv', filename1='proed_tweets_sample_3000.csv', filename2='prorec_tweets_sample_2000.csv')
    shuffle_csv(filename='reddit_processed.csv', columns=[],  new_filename='reddit_shuffled.csv')
    split_dataset_for_annotation(fromFile='reddit_shuffled.csv', rows_in_file=285, n_overlapping=30, new_filename1='reddit_data_frikk.csv', new_filename2='reddit_data_eirik.csv')
    
if __name__ == "__main__":
    print("Running main...")
    main()
