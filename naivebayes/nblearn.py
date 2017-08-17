import re
import sys

from naivebayes import helper as helper

unique_words = set()
word_dict = dict()
# word_dict = {
# 	positive: {
# 		word1: count
# 		word2: count
# 	}
# 	negative: {
# 		word1: count
# 		word2: count
# 	}
# }
count_dict = {'word': {}, 'review': {}}
# count_dict:{
#
# 	positive:
# 	negative:
# 	deceptive:
# 	truthful:
# }
data = dict()
# data{
# 	id:{
# 		sentence:
# 		class1:
# 		class2:
# 	}
# }
model = dict()

vocab = dict()


def add_to_word_dict(class_spec, parsed_word):
    """
    Adds the parsed word to the dictionary and initializes the count of all 4 classes for that word to 0.
    :param class_spec: 
    :param parsed_word: 
    :return: None
    """
    if parsed_word not in word_dict:
        word_dict[parsed_word] = {helper.POSITIVE: 0, helper.NEGATIVE: 0, helper.DECEPTIVE: 0, helper.TRUTHFUL: 0}
    word_dict[parsed_word][class_spec] += 1


def update_count_dict(class_spec, type):
    """
    Update the count of the word encountered to the corresponding class to which it belongs.
    :param class_spec: 
    :param type: 
    :return: None
    """
    if type == 'word':
        if class_spec in count_dict['word']:
            count_dict['word'][class_spec] += 1
        else:
            count_dict['word'][class_spec] = 1
    else:
        if class_spec in count_dict['review']:
            count_dict['review'][class_spec] += 1
        else:
            count_dict['review'][class_spec] = 1


def add_to_vocab(word):
    """
    Adds the word to the vocabulary of all words
    :param word: 
    :return: None
    """
    if word in vocab:
        vocab[word] += 1
    else:
        vocab[word] = 1


def compute_probability():
    """
    Compute the posterior probability for each word after applying laplace smoothing for words whose occurrence is None. 
    The 'posterior' object of the model is updated with the probability value for each word.
    :return: None
    """
    smoothing_factor = len(unique_words)
    model['posterior'] = []
    for word, word_obj in word_dict.items():
        for clas, count in word_obj.items():
            # Add 1 laplace transform smoothing
            count = float(count)
            prob_word = (count + 1) / (count_dict['word'][clas] + smoothing_factor)
            # put the values into model json
            prob_obj = {
                'word': word,
                'class': clas,
                'probability': prob_word
            }
            model['posterior'].append(prob_obj)


def compute_prior():
    """
    Compute the prior probability for each of 4 classes and update the model object.
    :return: None
    """
    # calculate prior probability for each class
    # P(prior) = count(positive)/ count(positive) + count(negative)
    positive = count_dict['review'][helper.POSITIVE] / (
        count_dict['review'][helper.NEGATIVE] + count_dict['review'][helper.POSITIVE])
    negative = count_dict['review'][helper.NEGATIVE] / (
        count_dict['review'][helper.POSITIVE] + count_dict['review'][helper.NEGATIVE])
    deceptive = count_dict['review'][helper.DECEPTIVE] / (
        count_dict['review'][helper.DECEPTIVE] + count_dict['review'][helper.TRUTHFUL])
    truthful = count_dict['review'][helper.TRUTHFUL] / (
        count_dict['review'][helper.TRUTHFUL] + count_dict['review'][helper.DECEPTIVE])

    prior = {
        helper.POSITIVE: positive,
        helper.NEGATIVE: negative,
        helper.DECEPTIVE: deceptive,
        helper.TRUTHFUL: truthful
    }
    model['prior'] = prior


def build_model():
    """
    Build the model object with the prior and posterior probability for each word encountered in the text.
    :return: 
    """
    for index, value in data.items():
        content = re.split(helper.get_regex(), value['review'])
        # clean up review text
        class_spec_1 = value['class1']
        class_spec_2 = value['class2']
        for word in content:
            parsed_word = helper.clean_up(word)
            if parsed_word and parsed_word != '':
                add_to_vocab(parsed_word)
                unique_words.add(parsed_word)
                # build separate classifier
                add_to_word_dict(class_spec_1, parsed_word)
                add_to_word_dict(class_spec_2, parsed_word)
                update_count_dict(class_spec_1, 'word')
                update_count_dict(class_spec_2, 'word')
        update_count_dict(class_spec_1, 'review')
        update_count_dict(class_spec_2, 'review')
    helper.write_file('unique_words.txt', unique_words)
    helper.write_json('worddict.json', word_dict)
    helper.write_json('countdict.json', count_dict)
    helper.write_json('vocab.json', vocab)


def main():
    """
    Build the model and computes the probability using the Bayes Theorem for each word and writes the model data to the
    model.json file.
    :return: None
    """
    # train_file = 'data/train-text.txt' - path to the train input data file
    # label_file = 'data/train-labels.txt' - path to the train class labels
    if len(sys.argv) == 3:
        train_file = sys.argv[1]
        label_file = sys.argv[2]

    with open(train_file) as train:
        for line in train:
            if line:
                index, content = line.split(' ', 1)
                index = helper.helper.clean_text(index)
                content = helper.clean_text(content)
                if index not in data:
                    data[index] = {}
                    data[index]['review'] = content

    with open(label_file) as label:
        for line in label:
            if line:
                index, class1_text, class2_text = line.split(' ')
                index = helper.clean_text(index)
                class1_text = helper.clean_text(class1_text)
                class2_text = helper.clean_text(class2_text)
                if index in data:
                    data[index]['class1'] = class1_text
                    data[index]['class2'] = class2_text
                else:
                    print('id not found')

    build_model()
    compute_probability()
    compute_prior()
    helper.write_json('model.json', model)


if __name__ == '__main__':
    main()
