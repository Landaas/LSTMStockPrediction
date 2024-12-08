
import statistics
from sentiment import sentiment
import os
import json
import re

def clean_tweet(tweet):
    # Remove RT at the start
    tweet = re.sub(r'RT\s*@\S+: ?', '', tweet)

    # Remove mentions (e.g., @username)
    tweet = re.sub(r'[@\$#]\w+', '', tweet)

    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    
    tweet = re.sub(r'(?<!\w)[^\w\s\']{2,}(?!\w)', '', tweet)

    # Remove extra spaces and leading/trailing whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    return tweet

def write_dict(dict):
    with open('result.json', 'w') as fp:
        json.dump(dict, fp)



def process_files(base_directory):
    result_dict = {}
    for root, _, files in os.walk(base_directory):
        dir = os.path.basename(os.path.normpath(root))
        result_dict[dir] = {}
        for file in files:
            file_path = os.path.join(root, file)
            scores = []
            count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, start=1):
                        try:
                            data = json.loads(line.strip())
                            if isinstance(data, dict) and "text" in data:
                                text = clean_tweet(data['text'])
                                scores.append(sentiment(text))
                                count += 1
                                
                        except json.JSONDecodeError:
                            print(f"Invalid JSON in file: {file_path} (Line {line_number})")
                    result_dict[dir][file] = (statistics.median(scores), count)
                
            except (OSError, IOError) as e:
                print(f"Error opening file: {file_path}. Error: {e}")
        write_dict(result_dict)
        print(f'{dir} complete')

base_directory = "tweet/raw"
process_files(base_directory)
