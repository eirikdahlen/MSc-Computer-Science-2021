import pandas as pd
from edit_datasets import read_jsonl
import csv


def individual_dist(path, dupes, dist):
    data = read_jsonl(path)
    if len(dist) == 2: 
        data = make_list_of_list_binary(data)
    for i in data:
        if i[0] in dupes.keys():
            dist[i[1]] += 1
    return dist


def find_duplicates(filename, isBinary):
    df = {}
    if isBinary:
        df = make_dataset_binary(filename) 
    else:
        df = pd.read_csv(filename, names=['text', 'label'])
    d = {}
    for index, row in df.iterrows():
        x = row['text']
        if x not in d:
            d[x] = [row['label']]
        else:
            d[x].append(row['label'])
    dupes = {}
    for key in d:
        if len(d[key]) > 1:
            dupes[key] = d[key]
    return dupes

def find_relative_agreement(d):
    length = len(d)
    diff = 0
    for key in d:
        if d[key][0] != d[key][1]:
            diff += 1
    score = (length-diff)/length
    return score, length

def cohen_kappa(dist1, dist2, relative_agreement, total):
    pe = 0
    for key in dist1:
        pe += dist1[key] * dist2[key]
    pe = pe / total**2
    k = (relative_agreement-pe)/(1-pe)
    print('Cohen kappa: ' + str(k))
    
def find_differences(d):
    dupes = {}
    for key in d:
        if (d.get(key)[0] != d.get(key)[1]):
            dupes[key] = d.get(key)
    return dupes    

def dict_to_csv(d, filename):
    columns = ['text', 'a1', 'a2']
  
    l = []
    for key in d.keys():
        l.append([key, d.get(key)[0], d.get(key)[1]])
    df = pd.DataFrame(l, columns=[columns])
    df.to_csv(filename)

def make_dataset_binary(filename):
    df = pd.read_csv(filename, names=['text', 'label'])
    for index, row in df.iterrows():
        if row['label'] == 'prorecovery':
            row['label'] = 'unrelated'
    return df 

def make_list_of_list_binary(l):
    for item in l:
        if item[1] == 'prorecovery':
            item[1] = 'unrelated'
    return l

def multi_label():
    multi_dist = {'proED': 0, 'prorecovery': 0, 'unrelated': 0}

    print('Multiclass: ')
    dup = find_duplicates('./labels.csv', False)
    score, total_dupes = find_relative_agreement(dup)

    print('Accuracy: ' + str(score))
    eirik = individual_dist('./data_twitter/labels/eirik.jsonl', dup, multi_dist)
    
    multi_dist = {'proED': 0, 'prorecovery': 0, 'unrelated': 0}
    frikk = individual_dist('./data_twitter/labels/frikk.jsonl', dup, multi_dist)

    cohen_kappa(eirik, frikk, score, total_dupes)

def binary_label():
    binary_dist = {'proED': 0, 'unrelated': 0}

    print('\n' + 'Binaryclass: ')
    dup = find_duplicates('./labels.csv', True)
    score, total_dupes = find_relative_agreement(dup)
    
    print('Accuracy: ' + str(score))
    eirik = individual_dist('./data_twitter/labels/eirik.jsonl', dup, binary_dist)
    
    binary_dist = {'proED': 0, 'unrelated': 0}
    frikk = individual_dist('./data_twitter/labels/eirik.jsonl', dup, binary_dist)

    cohen_kappa(eirik, frikk, score, total_dupes)

multi_label()
binary_label()

# dup = find_duplicates('./labels_reddit.csv', False)
# dict_to_csv(dup, 'dupes_reddit.csv')
# diff = find_differences(dup)
# dict_to_csv(diff, 'diff_reddit.csv')