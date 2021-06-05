import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np  
import re
from edit_datasets import write_csv
import matplotlib.ticker as mtick
import math
import emoji
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from matplotlib.font_manager import FontProperties
import mplcairo
matplotlib.use("module://mplcairo.qt")



def get_dataframe(filename,labels):
    df = pd.read_csv(filename,names=labels)
    return df

def label_distribution(df):
    nr_of_datapoints = len(df.index)
    dist = df.groupby('label').count() / nr_of_datapoints*100
    print(dist)

def string_distribution(df, string):
    dist = df[(df['text'].str.contains(string))].groupby('label').count() / df.groupby('label').count()
    return dist 

def word_count_dist(df):
    new_list = []
    for index, row in df.iterrows():
        label = row['label']
        twt = re.sub('[^A-Za-z0-9 ]+', '', row['text']).split()
        length = len(twt)
        if (length > 1000):
            continue
        new_list.append([label, length])
    new_df = pd.DataFrame(new_list, columns=['label', 'word_count'])
    dist = new_df.groupby('label')['word_count'].value_counts()
    print(new_df.groupby('label')['word_count'].mean())
    print(new_df['word_count'].mean())
    visualize_word_reddit(dist)

def replace_emojis(s):
    text = s.split()
    for i in range(len(text)):
        if 'EMOJI' in text[i]:
            text[i] = 'e'
    text = " ".join(text)
    return text

def length_distribution1(df):
    new_list = []
    for index, row in df.iterrows():
        label = row['label']
        text = replace_emojis(row['text'])
        length = len(text)
        if ((length < 5) or (length > 4000)):
            continue
        new_list.append([label, length])
    new_df = pd.DataFrame(new_list, columns=['label', 'string length'])
    dist = new_df.groupby('label')['string length'].value_counts()
    print(new_df.groupby('label')['string length'].mean())
    print(new_df['string length'].mean())
    visualize_char_reddit(dist)

def group_length_dist3(series, nr):
    new_list= [0]*nr
    check = 0
    total = series.sum()
    for index, value in series.items():
        new_list[math.floor(index/10) -1 ] += value
    final = [x / total for x in new_list]
    return final

def series_to_nested_list_write_file(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    labels = list(set(list(proEDs.index) + list(prorecs.index) + list(unrelateds.index)))

    proED = series_to_list(proEDs, len(labels))
    prorec = series_to_list(prorecs, len(labels))
    unrelated = series_to_list(unrelateds, len(labels))

    final_list = [proED, prorec, unrelated]
    print(final_list)
    write_csv(final_list, 'word_count_dist.csv')

def visualize_word(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    width = 0.3
    labels = list(set(list(proEDs.index) + list(prorecs.index) + list(unrelateds.index)))
    proED = series_to_list(proEDs, len(labels))
    prorec = series_to_list(prorecs, len(labels))
    unrelated = series_to_list(unrelateds, len(labels))

    r2 = np.arange(len(labels))
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(labels)

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i % 5)-4 != 0]

    ax.set_xlabel('Number of words')
    ax.set_ylabel('Percentage')

    ax.legend()

    fig.tight_layout()
    #plt.savefig('word_length_distribution.png')
    plt.show()

def visualize_char(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    labels = []
    proED, prorec, unrelated = '', '', ''
    width = 3

    length = 30
    proED = group_length_dist3(proEDs, length)
    prorec = group_length_dist3(prorecs, length)
    unrelated = group_length_dist3(unrelateds, length)
    labels = np.arange(10, length*10+1, 10).tolist()
    
    r2 = labels
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()


    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(labels)

    ax.set_xlabel('Number of characters')
    ax.set_ylabel('Percentage')

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()

    fig.tight_layout()
    #plt.savefig('char_length_distribution.png')
    plt.show()

def series_to_list(series, nr):
    new_list= [0]*nr
    total = series.sum()
    for i in range (len(new_list)):
        if series.get(i):
            new_list[i] = series.get(i)/total
    return new_list

def dist_twitter_terms(df_data):
    emoji = string_distribution(df_data, 'EMOJI')
    mention = string_distribution(df_data, 'MENTION')
    url = string_distribution(df_data, 'URL')
    proED = [emoji.loc['proED']['text'], mention.loc['proED']['text'], url.loc['proED']['text']]
    prorec = [emoji.loc['prorecovery']['text'], mention.loc['prorecovery']['text'], url.loc['prorecovery']['text']]
    unrelated = [emoji.loc['unrelated']['text'], mention.loc['unrelated']['text'], url.loc['unrelated']['text']]
    
    # proED = [emoji.loc['proED']['text'], url.loc['proED']['text']]
    # prorec = [emoji.loc['prorecovery']['text'], url.loc['prorecovery']['text']]
    # unrelated = [emoji.loc['unrelated']['text'], url.loc['unrelated']['text']]

    # print(emoji)
    # print(url)
    # print(proED)
    # print(prorec)
    # print(unrelated) 

    labels = ['EMOJI', 'MENTION', 'URL']
    #labels = ['EMOJI','URL']

    width = 0.3

    r2 = np.arange(3)
    #r2 = np.arange(2)
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks([0,1,2])
    #ax.set_xticks([0,1])

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Internet term')
    ax.set_ylabel('Percentage')

    ax.legend()

    fig.tight_layout()
    plt.show()

def get_most_popular_emojis(df):
    emojis = {}
    for index, row in df.iterrows():
        text = row['text'].split()
        for item in text:
            if 'SkinTone' in item:
                continue
            if item.find('EMOJI') != -1:
                if item in emojis.keys():
                    emojis[item] += 1
                else:
                    emojis[item] = 1
    print(emojis)
    most_popular = sorted(emojis, key=emojis.get, reverse=True)[:6]
    print(most_popular)
    return most_popular

def popular_emojis(df):
    proED = []
    prorec = []
    unrelated = []
    most_popular = get_most_popular_emojis(df)
    for emojii in most_popular:
        e = string_distribution(df, emojii)
        print(e)
        proED.append(e.loc['proED']['text'])
        prorec.append(e.loc['prorecovery']['text'])
        unrelated.append(e.loc['unrelated']['text'])
    
    #most_popular = [':red_heart:', ':loudly_crying_face:', ':face_with_tears_of_joy:',':folded_hands:', ':sparkles:', ':clapping_hands:']
    most_popular_twitter = ['EMOJIRedHeart', 'EMOJILoudlyCryingFace', 'EMOJIFaceWithTearsOfJoy', 'EMOJIFoldedHands', 'EMOJISparkles', 'EMOJIClappingHands']
    #most_popular_reddit = ['EMOJIRedHeart', 'EMOJIFire', 'EMOJIRocket', 'EMOJIFemaleSign', 'EMOJILightSkinTone', 'EMOJIMediumLightSkinTone']
    most_popular_reddit = [':red_heart:', ':fire:', ':rocket:', ':female_sign:', ':male_sign:', ':rainbow:']
    twitter_emojis = ['❤️', '😭', '😂', '🙏' ,'✨', '👏']
    
    prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')

    width = 0.3

    r2 = np.arange(6)
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(np.arange(6))

    ax.set_xticklabels(twitter_emojis, fontproperties=prop)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel('Emoji')
    ax.set_ylabel('Percentage')

    ax.legend()

    fig.tight_layout()
    plt.show()

def compute_sentiment(s):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(s)['compound']
    if sentiment >= 0.05:
        return 'pos'
    elif sentiment <= -0.05:
        return 'neg'
    else:
        return 'neu'

def sentiment_analysis(df): 
    new_list = []
    for index, row in df.iterrows():
        text = row['text']
        label = row['label']
        sentiment = compute_sentiment(text)
        new_list.append([label, sentiment])
    new_df = pd.DataFrame(new_list, columns=['label', 'sentiment'])
    dist = new_df.groupby('label')['sentiment'].value_counts()
    print(new_df.groupby('label')['sentiment'].value_counts())
    visualize_sentiment(dist)

def visualize_sentiment(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    width = 0.3
    labels = ['Positive', ' Neutral', 'Negative']

    proED = series_to_list(proEDs, len(labels))
    prorec = series_to_list(prorecs, len(labels))
    unrelated = series_to_list(unrelateds, len(labels))

    print(proED)
    print(prorec)
    print(unrelated)

    r2 = np.arange(len(labels))
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(r2)

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Percentage')

    ax.legend()

    fig.tight_layout()
    #plt.savefig('sentiment_distribution.png')
    plt.show()

def group_length_dist50(series, nr):
    new_list= [0]*nr
    check = 0
    total = series.sum()
    for index, value in series.items():
        new_list[math.floor(index/100) ] += value
    final = [x / total for x in new_list[:-1]]
    return final

def visualize_word_reddit(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    labels = []
    proED, prorec, unrelated = '', '', ''
    width = 20

    length = 10
    proED = group_length_dist50(proEDs, length)
    prorec = group_length_dist50(prorecs, length)
    unrelated = group_length_dist50(unrelateds, length)
    labels = np.arange(100, length*100+1, 100).tolist()
    
    r2 = labels[:-1]
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(labels)

    ax.set_xlabel('Number of words')
    ax.set_ylabel('Percentage')

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()

    #[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if ((i+1) % 5) != 0]


    fig.tight_layout()
    #plt.savefig('char_length_distribution.png')
    plt.show()

def group_length_dist100(series, nr):
    new_list= [0]*nr
    check = 0
    total = series.sum()
    for index, value in series.items():
        new_list[math.floor(index/100) - 1 ] += value
    final = [x / total for x in new_list[:-1]]
    return final

def visualize_char_reddit(series):
    proEDs = series['proED']
    prorecs = series['prorecovery']
    unrelateds = series['unrelated']
    labels = []
    proED, prorec, unrelated = '', '', ''
    width = 20


    length = 40
    proED = group_length_dist50(proEDs, length)
    prorec = group_length_dist50(prorecs, length)
    unrelated = group_length_dist50(unrelateds, length)
    labels = np.arange(100, length*100+1, 100).tolist()


    r2 = labels[:-1]
    r1 = [x - width for x in r2]
    r3 = [x + width for x in r2]

    fig, ax = plt.subplots()

    rects1 = ax.bar(r1, proED, width, edgecolor='white', label='proED')
    rects2 = ax.bar(r2, prorec, width, edgecolor='white', label='prorecovery')
    rects3 = ax.bar(r3, unrelated, width, edgecolor='white', label='unrelated')

    ax.set_xticks(labels)

    ax.set_xlabel('Number of characters')
    ax.set_ylabel('Percentage')

    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()

    [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if ((i+1) % 5) != 0]


    fig.tight_layout()
    #plt.savefig('char_length_distribution.png')
    plt.show()

    

labels_twitter = ['tweet_id','user_id','name','screen_name','description','text','label']
labels_reddit = ['post_id', 'subreddit', 'text', 'label']


df_data = get_dataframe('./smack.csv', labels=labels_twitter)
df_data = df_data[df_data['label'] != 'label']
#sentiment_analysis(df_data)


#label_distribution(df_data)
#word_count_dist(df_data)
#length_distribution1(df_data)
#dist_twitter_terms(df_data)
popular_emojis(df_data)