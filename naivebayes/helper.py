import sys
import json

POSITIVE = 'positive'
NEGATIVE = 'negative'
DECEPTIVE = 'deceptive'
TRUTHFUL = 'truthful'

config = {
    'lowercase': True,
    'stopwords': False,
    'symbols': False,
    'strip': True,
    'ignore_numbers': False,
    'classifier': [
        POSITIVE,
        NEGATIVE,
        DECEPTIVE,
        TRUTHFUL
    ]
}


def get_stopwords():
    stop = []
    with open('./stop-words.txt') as stop_f:
        for line in stop_f:
            stop.append(line.strip())
    stop_f.close()
    return stop


stop_words = get_stopwords()


def get_regex():
    return r"[,:;#^&*/\"()\.\- ]+"


def get_symbols():
    return ['`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
            '-', '_', '+', '=', '{', '}', '[', ']', ';', ':', '\'', '"',
            '<', '>', ',', '.', '/', '?']


def clean_text(text):
    return text.replace('\n', '').replace('\t', '')


def clean_up(input_word):
    parsed_word = input_word.strip()
    if config['symbols'] and parsed_word:
        if parsed_word in get_symbols():
            parsed_word = ''
    if config['lowercase'] and parsed_word:
        parsed_word = parsed_word.lower()
    if config['stopwords'] and parsed_word:
        if parsed_word in stop_words:
            parsed_word = ''
    if config['strip'] and parsed_word:
        parsed_word = parsed_word.strip('\'')
    if config['ignore_numbers'] and parsed_word:
        if parsed_word.isdigit():
            parsed_word = ''
    return parsed_word


def write_json(file_name, data):
    with open('./' + file_name, 'w') as outfile:
        json.dump(data, outfile)
    outfile.close()


def write_file(file_name, data):
    with open('results/' + file_name, 'w') as outfile:
        for value in data:
            outfile.write(value + '\n')
    outfile.close()
