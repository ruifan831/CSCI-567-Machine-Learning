import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    word2idx = dict(zip(unique_words.keys(),range(len(unique_words))))
    tag2idx = dict(zip(tags,range(len(tags))))


    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    total = sum(unique_words.values())
    for line in train_data:
        for word,tag in list(zip(line.words,line.tags)):
            pi[tag2idx[tag]]+=1
            B[tag2idx[tag],word2idx[word]]+=1
        for i in range(len(line.tags)-1):
            cur_tag = tag2idx[line.tags[i]]
            next_tag = tag2idx[line.tags[i+1]]
            A[cur_tag,next_tag] +=1
    for index,total in enumerate(np.sum(A,axis=1)):
        if total != 0:
            A[index,:] = A[index,:]/total
    for index,total in enumerate(np.sum(B,axis=1)):
        if total != 0:
            B[index,:] = B[index,:]/total
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict:
                index = len(model.obs_dict)
                model.obs_dict[word] = index
                addition = np.full((len(model.B),1),1e-6)
                model.B = np.hstack((model.B,addition))
        tagging.append(model.viterbi(line.words))

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
