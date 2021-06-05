"""
Steps:
1. Remove whitespaces
2. Lowercase everything
3. Placeholder for MENTION, URL
4. Handle EMOJI's
5. Character entity references (gt, lt, &amp)
6. Remove punctation and numbers
7. SoMe abbrevations (ppl, lmk, idk, wtf, lol, pls, abt etc..)
8. Remove symbols/special tokens
9. Remove non-English tweets
10. Remove tweets with short length
"""
import re
import emoji as emoji_handler
import preprocessor as twp
import string

def remove_whitespace(post):
    """
    Remove whitespaces in one datapoint instance in the dataset.
    The format is: [post_id, user_id, user_name, screen_name, user_description, tweet_text]

    Args:
        post ([list]): [list of strings with information about post]
    """
    
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    user_description = user_description.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    tweet_text = tweet_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    return [post_id, user_id, user_name.strip(), screen_name.strip(), user_description.strip(), tweet_text.strip()]

def to_lowercase(post):
    return [text_field.lower() for text_field in post]

def handle_punctuation_and_numbers(post):
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    remove = '"$%&()*+/:;<=>@[\\]^_`{|}~' + string.digits
    user_name = user_name.translate(user_name.maketrans('', '', remove))
    user_description = user_description.translate(user_description.maketrans('', '', remove))
    tweet_text = tweet_text.translate(tweet_text.maketrans('', '', remove))
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]
    
def replace_from_regex(regex, text_field, replace_string):
    match = re.findall(regex, text_field)
    if match:
        for match_tuple in match:
            if replace_string == 'URL':
                match_tuple = match_tuple[0]
            text_field = text_field.replace(match_tuple, replace_string)
    return text_field

def handle_urls(post):
    regex = '(([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#\.]?[\w-]+)*\/?)'
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    user_description = replace_from_regex(regex, user_description, 'URL')
    tweet_text = replace_from_regex(regex, tweet_text, 'URL')
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]
    

def handle_mentions(post):
    regex = '((?<=^|(?<=[^a-zA-Z0-9-_\.]))@[A-Za-z0-9-_]+(?=[^a-zA-Z0-9-_\.]))' # This regex avoids match on emails etc
    post[-1] = replace_from_regex(regex, post[-1], 'MENTION')
    return post

def handle_emoji(post):
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    
    parsed_description = twp.parse(user_description)
    parsed_tweet = twp.parse(tweet_text)
    
    if parsed_description.emojis:
        for emoji in parsed_description.emojis:
            emoji_replacement_string = parse_emoji_to_string(emoji_handler.demojize(emoji.match))
            user_description = user_description.replace(emoji.match, emoji_replacement_string)
    
    if parsed_tweet.emojis:
        for emoji in parsed_tweet.emojis:
            emoji_replacement_string = parse_emoji_to_string(emoji_handler.demojize(emoji.match))
            tweet_text = tweet_text.replace(emoji.match, emoji_replacement_string)
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]

def parse_emoji_to_string(demojized_emoji):
    regex = ':([^ :]+):'
    match = re.search(regex, demojized_emoji)
    if not match:
        return ''
    match = match.group(1)
    match = match.replace('-', '_')
    match_list = match.split('_')
    match_list = [word.capitalize() for word in match_list]
    new_word = ' EMOJI' + ''.join(match_list) + ' '
    return new_word
        
def remove_non_ascii(post):
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    user_name = user_name.encode('ascii', errors='ignore').decode()
    user_description = user_description.encode('ascii', errors='ignore').decode()
    tweet_text = tweet_text.encode('ascii', errors='ignore').decode()
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]

def handle_character_entity_references(post):
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    user_description = re.sub(r"&gt;", ">", user_description)
    user_description = re.sub(r"&lt;", "<", user_description)
    user_description = re.sub(r"&amp;", "and", user_description) # Should've been "and"
    user_description = re.sub(r"<3*", " EMOJIRedHeart ", user_description)
    tweet_text = re.sub(r"&gt;", ">", tweet_text)
    tweet_text = re.sub(r"&lt;", "<", tweet_text)
    tweet_text = re.sub(r"&amp;", "and", tweet_text) # Should've been "and"
    tweet_text = re.sub(r"<3*", " EMOJIRedHeart ", tweet_text)
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]

def handle_abbreviations(post):
    post_id, user_id, user_name, screen_name, user_description, tweet_text = post
    user_description = re.sub(r"\bppl\b", "people", user_description)
    user_description = re.sub(r"\blmk\b", "let me know", user_description)
    user_description = re.sub(r"\bidk\b", "i don't know", user_description)
    user_description = re.sub(r"\bwtf\b", "what the fuck", user_description)
    user_description = re.sub(r"\bpls\b", "please", user_description)
    user_description = re.sub(r"\babt\b", "about", user_description)
    user_description = re.sub(r"\bfr\b", "for real", user_description)
    user_description = re.sub(r"\bty\b", "thank you", user_description)
    user_description = re.sub(r"\brly\b", "really", user_description)
    user_description = re.sub(r"\bu\b", "you", user_description)
    user_description = re.sub(r"\brn\b", "right now", user_description)
    user_description = re.sub(r"\bnvm\b", "never mind", user_description)
    tweet_text = re.sub(r"\bppl\b", "people", tweet_text)
    tweet_text = re.sub(r"\blmk\b", "let me know", tweet_text)
    tweet_text = re.sub(r"\bidk\b", "i don't know", tweet_text)
    tweet_text = re.sub(r"\bwtf\b", "what the fuck", tweet_text)
    tweet_text = re.sub(r"\bpls\b", "please", tweet_text)
    tweet_text = re.sub(r"\babt\b", "about", tweet_text)
    tweet_text = re.sub(r"\bfr\b", "for real", tweet_text)
    tweet_text = re.sub(r"\bty\b", "thank you", tweet_text)
    tweet_text = re.sub(r"\brly\b", "really", tweet_text)
    tweet_text = re.sub(r"\bu\b", "you", tweet_text)
    tweet_text = re.sub(r"\brn\b", "right now", tweet_text)
    tweet_text = re.sub(r"\bnvm\b", "never mind", tweet_text)
    return [post_id, user_id, user_name, screen_name, user_description, tweet_text]
