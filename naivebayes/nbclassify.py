import json
import math
import re
import sys

from naivebayes import helper as helper

model_path = './model.json'
output_path = 'data/nboutput.txt'

model = dict()
word_table = dict()
data = dict()
review_table = dict()
classifier = helper.config['classifier']


def decide_class():
    """
    Based on the probability value determined by the Bayes Theorem, the label is assigned for highest probability values
    among one of {POSITIVE, NEGATIVE} and one of {DECEPTIVE, TRUTHFUL}
    :return: None
    """
    with open(output_path, 'w') as f:
        for index, value in review_table.items():
            class1 = helper.TRUTHFUL
            class2 = helper.POSITIVE
            if value[helper.TRUTHFUL] < value[helper.DECEPTIVE]:
                class1 = helper.DECEPTIVE
            if value[helper.POSITIVE] < value[helper.NEGATIVE]:
                class2 = helper.NEGATIVE

            f.write('{} {} {}\n'.format(index, class1, class2))
    f.close()


def extract_words():
    """
    Extracts the words from the given sentences and creates a probability value for each of 4 classes for every
    review/sentence by summing the individual word probability.
    :param model:
    :return: None
    """
    for index, value in data.items():
        content = re.split(helper.get_regex(), value['review'])
        # clean up review text
        for clas in range(len(classifier)):
            clas_name = classifier[clas]
            prob_sum = math.log(model['prior'][clas_name])
            for word in content:
                parsed_word = helper.clean_up(word)
                if parsed_word in word_table:
                    clas_prob_map = word_table[parsed_word]
                    if clas_name in clas_prob_map:
                        prob_sum += math.log(clas_prob_map[clas_name])

            if index in review_table:
                review_table[index][clas_name] = prob_sum
            else:
                review_table[index] = {clas_name: prob_sum}
    helper.write_json('review_table.json', review_table)


def build_word_table():
    """
    For each word creates an object which contains the posterior probability value for all the 4 classes.
    :param model:
    :return: None
    """
    posterior = model['posterior']
    for index in range(len(posterior)):
        word = posterior[index]['word']
        word_class = posterior[index]['class']
        word_prob = posterior[index]['probability']
        if word in word_table:
            word_table[word][word_class] = word_prob
        else:
            word_table[word] = {word_class: word_prob}
    helper.write_json('classify.json', word_table)


def main():
    """
    Loads the test data file and the model file. Predicts the test class labels for every sentence/review.
    The result is written to the output file given in the output path variable
    :return:
    """
    # test_file = 'data/test_file' - path to the test file to test the Naives Bayes model
    if len(sys.argv) == 2:
        test_file = sys.argv[1]
    with open(test_file) as test:
        for line in test:
            if line:
                index, content = line.split(' ', 1)
                index = helper.clean_text(index)
                content = helper.clean_text(content)
                if not index in data:
                    data[index] = {}
                    data[index]['review'] = content

    with open(model_path) as f:
        model = json.load(f)
    # build the word table with all probabilities

    build_word_table()

    # extract the words form the text

    extract_words()

    # determine class

    decide_class()

    test.close()


if __name__ == '__main__':
    main()
