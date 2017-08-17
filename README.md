# NLP

## Naive Bayes 
A naive Bayes classifier to identify reviews as either truthful or deceptive, and either positive or negative. The word tokens are used as features for classification. The algorithm implementation can be used for other classification problems as well. 

## Data

The data will consist of two files.

1. A text file train-text.txt with a single training instance (review) per line. The first token in the each line is a unique 20-character alphanumeric identifier, which is followed by the text of the review.
2. A label file train-labels.txt with labels for the corresponding reviews. Each line consists of three tokens: a unique 20-character alphanumeric identifier corresponding to a review, a label truthful or deceptive, and a label positive or negative.

## Programs

There are two programs defined: nblearn.py will learn a naive Bayes model from the training data, and nbclassify.py will use the model to classify new data. The learning program will be invoked in the following way:

> python nblearn.py /path/to/text/file /path/to/label/file

The arguments are the two training files; the program will learn a naive Bayes model, and write the model parameters to a file called nbmodel.txt. The format of the model is up to you, but it should contain the model parameters (that is, the various probabilities).

The classification program will be invoked in the following way:

> python nbclassify.py /path/to/text/file

The argument is the test data file, which has the same format as the training text file. The program will read the parameters of a naive Bayes model from the file nbmodel.txt, classify each entry in the test data, and write the results to a text file called nboutput.txt in the same format as the label file from the training data.

