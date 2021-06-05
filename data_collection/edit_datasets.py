import json_lines
import csv
import numpy as np
import pandas as pd
import os

labels_twitter = ['tweet_id','user_id','name','screen_name','description','text','label']
labels_reddit = ['post_id', 'subreddit', 'text', 'label']

def extract_text(s):
    return s.split("[a-zA-Z]+")


def df_to_csv(dataframe, filename):
    dataframe.to_csv(filename)


def read_jsonl(path):
    dataset = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            if len(item['accept']) == 0:
                continue
            labeled_text = [item['text'], item['accept'][0]]
            dataset.append(labeled_text)
    return dataset

def read_all_jsonl_to_csv(dirname, new_filename):
    dataset = []
    for filename in os.listdir(dirname):
        relative_path = os.path.join(dirname, filename)
        data = read_jsonl(relative_path)
        for datapoint in data:
            dataset.append(datapoint) 
    write_csv(dataset, new_filename)

def write_csv(dataset, new_filename):
    with open(new_filename, 'w') as f:      
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(dataset) 

def merge_csv(dirname, new_filename):
    dataset = []
    for filename in os.listdir(dirname):
        relative_path = os.path.join(dirname, filename)
        df_data = pd.read_csv(relative_path)
        for index, row in df_data.iterrows():
            data_point = row.values.tolist()
            dataset.append(data_point)
    write_csv(dataset, new_filename) 

def merge_datasets(path_data, path_labeled, label_type):
    #Labels from the original dataset. Must be changed based on which SM.
    df_data = pd.read_csv(path_data, names=label_type[:-1])
    df_labels = pd.read_csv(path_labeled, names=['text', 'label'])
    headers = list(df_data.columns)
    headers.append('label')
    labeled_data = []
    for index, row in df_data.iterrows():
        label = df_labels.loc[df_labels['text'] == row['text']]['label']
        if len(label) != 0:
            data_point = row.values.tolist()
            data_point.append(label.values.tolist()[0])
            labeled_data.append(data_point)
    df = pd.DataFrame(labeled_data, columns=headers)
    return df
 
def make_dataset(dir_datasets, dir_labeled, fn_unlabeled_dataset, fn_labeled_csv, fn_final_dataset, label_type):
    read_all_jsonl_to_csv(dir_labeled,fn_labeled_csv)
    merge_csv(dir_datasets, fn_unlabeled_dataset)
    df_dataset = merge_datasets('./' + fn_unlabeled_dataset, './' + fn_labeled_csv, label_type)
    df_dataset = df_dataset.values.tolist()
    print(df_dataset)
    write_csv(df_dataset,fn_final_dataset)
    

def remove_duplicates(fn_dupes, fn_diff, fn_dataset, label_type, new_name):
    final_dataset = []
    dupes = pd.read_csv(fn_dupes)
    diff =  pd.read_csv(fn_diff, sep=';')
    df = pd.read_csv(fn_dataset, names=label_type)
    dup_list = []
    diff_list = []
    for index, row in df.iterrows():
        label_diff = diff.loc[diff['text'] == row['text']]['FINAL']
        if (len(label_diff) != 0 and row['text'] not in diff_list):
            diff_list.append(row['text'])
            data_point = []
            if len(label_type) == 7:
                data_point = [row['tweet_id'], row['user_id'], row['name'], row['screen_name'], row['description'], row['text']]
            else: 
                data_point = [row['post_id'], row['subreddit'], row['text']]
            data_point.append(list(label_diff)[0])
            final_dataset.append(data_point)
            continue
        label_dupes = dupes.loc[dupes['text'] == row['text']]['a1']
        if (len(label_dupes) != 0 and row['text'] not in dup_list and row['text'] not in diff_list):
            dup_list.append(row['text'])
            data_point = []
            if len(label_type) == 7:
                data_point = [row['tweet_id'], row['user_id'], row['name'], row['screen_name'], row['description'], row['text']]
            else: 
                data_point = [row['post_id'], row['subreddit'], row['text']]
            data_point.append(list(label_dupes)[0])
            final_dataset.append(data_point)
            continue
        if row['text'] not in dup_list and row['text'] not in diff_list: 
            row = list(row)
            final_dataset.append(row)
    write_csv(final_dataset, new_name)

def merge_with_unrelated(fn_targeted, fn_unrelated):
    df_targeted = pd.read_csv(fn_targeted, names=labels_reddit)
    unrelated = []
    df_unrelated = pd.read_csv(fn_unrelated)[:100]
    for index, row in df_unrelated.iterrows():
        unrelated.append([row['post_id'], row['subreddit'], row['text'], 'unrelated'])
    df_unrelated_labeled = pd.DataFrame(unrelated, columns=labels_reddit)
    df = pd.concat([df_targeted, df_unrelated_labeled])
    print(df)
    df.to_csv('smack_reddit.csv')
    


def twitter():
    make_dataset('./data_twitter/datasets', './data_twitter/labels','unlabeled_dataset.csv', 'labels.csv', 'final_dataset.csv', labels_twitter)
    remove_duplicates('./dupes.csv', './diff_solved.csv', './final_dataset.csv', labels_twitter, 'smack.csv')


def reddit():
    make_dataset('./data_reddit/datasets', './data_reddit/labels','unlabeled_reddit.csv', 'labels_reddit.csv', 'final_reddit.csv', labels_reddit)
    remove_duplicates('./dupes_reddit.csv', './diff_reddit_solved.csv', './final_reddit.csv', labels_reddit, 'smack_reddit.csv')
    merge_with_unrelated('./smack_reddit.csv', './reddit_processed_unrelated.csv')

#twitter()
#reddit()




def find_ignored(dirname):
    count = 0
    for filename in os.listdir(dirname):
        path = os.path.join(dirname, filename)
        
        with open(path, 'rb') as f:
            for item in json_lines.reader(f):
                if item['answer'] != 'accept':
                #if len(item['accept']) == 0:
                   count += 1
    print (count)

find_ignored('./data_twitter/labels')