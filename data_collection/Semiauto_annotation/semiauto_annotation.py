import pandas as pd
from Preprocessing.data_filtering import is_short_post, shuffle_csv

pd.options.mode.chained_assignment = None  # default='warn'


def concat_dataframes(dataframes):
    return pd.concat(dataframes)


def drop_existing_indices(ids, df):
    print(len(df))
    df_filtered = df[~df['tweet_id'].isin(ids)]
    print(len(df_filtered))
    print(f"Diff: {len(df) - len(df_filtered)}")
    return df_filtered


def semi_auto_annotation(df, proed_kws=[], prorec_kws=[]):
    print(df)
    new_data = []
    unrelated_counter = 0
    proed_counter = 0
    prorec_counter = 0
    for index, row in df.iterrows():
        if proed_counter > 25000 and prorec_counter > 12000:
            break
        if row['tweet_id'] == ' ' and is_short_post(row['text']):
            continue
        label = ''
        for kw in proed_kws:
            if kw in row['text']:
                if proed_counter > 25000:
                    continue
                label = 'proED'
                proed_counter += 1
                break
        if not label:
            for kw in prorec_kws:
                if kw in row['text']:
                    label = 'prorecovery'
                    prorec_counter += 1
                    break
        if not label:
            if unrelated_counter >= 100000:
                continue
            label = 'unrelated'
            unrelated_counter += 1
        data_point = row.values.tolist()
        data_point.append(label)
        new_data.append(data_point)
    new_df = pd.DataFrame(new_data, columns=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text',
                                             'label'])
    print(new_df)
    return new_df


def main():
    annotated_dataset = pd.read_csv('../smack.csv',
                                    names=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text',
                                           'label'])
    print(len(annotated_dataset))
    ids = set(annotated_dataset['tweet_id'])
    print(len(ids))

    proed_data = pd.read_csv('../twitter_data_proed_filtered.csv')
    proed_data = proed_data[['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text']]
    prorec_data = pd.read_csv('../twitter_data_prorec_filtered.csv')
    prorec_data = prorec_data[['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text']]
    ng_data = pd.read_csv('andrea_martine_dataset_final.csv',
                          names=['name', 'screen_name', 'timestamp', 'description', 'location', 'text', 'label'])
    ng_data = ng_data[['name', 'screen_name', 'description', 'text', 'label']]
    ng_data.insert(loc=0, column='tweet_id', value=' ')
    ng_data.insert(loc=1, column='user_id', value=' ')
    ng_data = ng_data.sample(frac=1)
    proed_ng = ng_data.loc[ng_data['label'] == 'pro_ed']
    prorec_ng = ng_data.loc[ng_data['label'] == 'pro_recovery']
    proed_ng = proed_ng.drop(columns=['label'])
    prorec_ng = prorec_ng.drop(columns=['label'])

    print(len(proed_data), len(prorec_data), int(len(proed_data) + len(prorec_data)))
    full_df = concat_dataframes(dataframes=[proed_data, prorec_data, proed_ng, prorec_ng])
    df = drop_existing_indices(ids=ids, df=full_df)
    # semi_annotated_data = semi_auto_annotation(df,
    #                                            proed_kws=['proana', 'thinspo', 'bonespo', 'edtwt', 'proed',
    #                                                       'thinspiration', 'thinsp0', 'anatwt', 'thighgap', 'promia',
    #                                                       'ricecaketwt', 'skinny', 'meanspo', 'pro ana', 'pro ed',
    #                                                       'pro mia', 'thygap', 'ugw', 'edproblems', ' tw '],
    #                                            prorec_kws=['edaw', 'nedaw', 'edrecovery', 'recoverywarriors', 'beated',
    #                                                        'eatingdisorderrecovery', 'anorexiarecovery', 'anarecovery',
    #                                                        'nedwareness', 'eatingdisordersupport', 'ed recovery',
    #                                                        'recover from eating', 'support eating disorder recovery',
    #                                                        'ed awareness'])
    # proed = semi_annotated_data['label'].apply(lambda label: True if label == 'proED' else False)
    # print(len(proed[proed == True].index))
    # prorec = semi_annotated_data['label'].apply(lambda label: True if label == 'prorecovery' else False)
    # print(len(prorec[prorec == True].index))
    # unrelated = semi_annotated_data['label'].apply(lambda label: True if label == 'unrelated' else False)
    # print(len(unrelated[unrelated == True].index))
    #semi_annotated_data.to_csv('semi_annotated_data_twitter.csv', index=False)


if __name__ == "__main__":
    print("Running main...")
    #shuffle_csv(filename='semi_annotated_data_twitter.csv', columns=['tweet_id', 'user_id', 'name', 'screen_name', 'description', 'text', 'label'], new_filename='semi_auto_shuffled.csv')

    main()
