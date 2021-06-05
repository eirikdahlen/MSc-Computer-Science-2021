def get_txt(path):
    f = open(path, 'r')
    return f


def write_txt(path, text):
    f = open(path, 'w')
    f.write(text)
    f.close()

def get_tao_tags():
    f = get_txt('Tags/tao_tags.txt')
    tags = []
    for i in f:
        #print(i)
        if i.startswith('#'):
            continue
        i= i.split()
        if float(i[1]) > 500:
            tags.append(i[0])
    return tags
    
def get_tags_from_txt(path):
    f = get_txt(path)
    tags = f.read().splitlines()
    return tags

def remove_hashtags(path):
    f = get_txt(path)
    tags_without_hashtag = []
    for i in f:
        if i.startswith('#'):
            i = i.replace('#', '')
        i = i.replace('\n', '')
        tags_without_hashtag.append(i)
    return tags_without_hashtag
    

def create_codebook():
    gieaver_tags = set(get_tags_from_txt('Tags/gieaver_tags.txt'))
    own_tags = set(get_tags_from_txt('Tags/own_proED_tags.txt'))
    tao_tags = set(get_tao_tags())
    all_tags = gieaver_tags.union(tao_tags).union(own_tags)
    tags_as_string = '\n'.join(map(str, all_tags))
    write_txt('Tags/tags.txt', tags_as_string)
    return tags_as_string

def get_tags_to_search(filename):
    with open(filename, 'r') as file:
        return ['#' + line.strip() for line in file.readlines()]
    
def create_unrelated_tags():
    unrelated_tags = remove_hashtags('Tags/unrelated_raw.txt')
    unrelated_tags_as_string = '\n'.join(map(str, unrelated_tags))
    write_txt('Tags/unreleated.txt', unrelated_tags_as_string)
    return unrelated_tags_as_string

"""
t = create_codebook()
print(t)
"""