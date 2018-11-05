# coding=utf-8
import os
import pickle
import pandas as pd
import numpy as np
from my_code.classifier import classify

user_message_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_message_network.txt"
message_word_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/message_word_network.txt"
user_word_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_word_network.txt"

outputemb = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/outputemb.txt"
outputctx = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/outputctx.txt"
binary = 0
size = 100
negative = 5
mw_samples = 1000
uw_samples = 500
um_samples = 100
rho = 0.025
iter = 5
thread = 40

hybird_topic_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/hybird_topic_user_emb.pk"
hybird_word_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/hybird_word_user_emb.pk"
shallow_topic_message_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/shallow_topic_message_emb.pk"
shallow_topic_user_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/shallow_topic_user_emb.pk"
word_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/word_emb.pk"
word_user_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/word_user_emb.pk"

user_message_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_message_relation.csv"
message_word_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/message_word_relation.csv"
user_word_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_word_relation.csv"

user_vocab_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_vocab.csv"


def train_network():

    command = "g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result hmu.cpp -o hmu -lgsl -lm -lgslcblas"
    print("============= compile hmu ====================")
    print(command)
    os.system(command)

    command = "./hmu "
    command += "-trainmw %s " % (message_word_network_path)
    command += "-trainuw %s " % (user_word_network_path)
    command += "-trainum %s " % (user_message_network_path)
    command += "-outputemb %s " % (outputemb)
    command += "-outputctx %s " % (outputctx)
    command += "-binary %s " % (binary)
    command += "-size %s " % (size)
    command += "-negative %s " % (negative)
    command += "-mw_samples %s " % (mw_samples)
    command += "-uw_samples %s " % (uw_samples)
    command += "-um_samples %s " % (um_samples)
    command += "-rho %s " % (rho)
    command += "-iter %s " % (iter)
    command += "-threads %s " % (thread)
    print("====================== train hmu ======================")
    print(command)
    os.system(command)
    print('training is over!')

train_network()


def save_hybrid_topic_user_emb():
    lines = open(outputemb).readlines()
    number = len(lines)
    user_vocab = {}
    for i in range (1, number, 1):
        tokens = lines[i].strip().split(" ")
        if 'U' in tokens[0]:
            user = tokens[0]
            vectors = [float(v) for v in tokens[1:]]
            user_vocab[user] = vectors
    print(len(user_vocab))
    with open(hybird_topic_path, 'wb') as g:
        pickle.dump(user_vocab, g)

def save_hybird_word_user_emb():
    lines = open(outputctx).readlines() # ctx
    number = len(lines)
    user_vocab = {}
    for i in range(1, number, 1):
        tokens = lines[i].strip().split(" ")
        if 'U' in tokens[0]:
            user = tokens[0]
            vectors = [float(v) for v in tokens[1:]]
            user_vocab[user] = vectors
    print(len(user_vocab))
    with open(hybird_word_path, 'wb') as g:
        pickle.dump(user_vocab, g)

def save_shallow_topic_emb():
    lines = open(outputemb).readlines()
    number = len(lines)
    message_vocab = {}
    for i in range(1, number, 1):
        tokens = lines[i].strip().split(" ")
        if 'M' in tokens[0]:
            message = tokens[0]
            vectors = [float(v) for v in tokens[1:]]
            message_vocab[message] = vectors
    print(len(message_vocab))
    with open(shallow_topic_message_path, 'wb') as g:
        pickle.dump(message_vocab, g)

def save_word_emb():
    lines = open(outputemb).readlines()
    number = len(lines)
    word_vocab = {}
    for i in range(1, number, 1):
        tokens = lines[i].strip().split(" ")
        if 'W' in tokens[0]:
            word = tokens[0]
            vectors = [float(v) for v in tokens[1:]]
            word_vocab[word] = vectors
    print(len(word_vocab))
    with open(word_path, 'wb') as g:
        pickle.dump(word_vocab, g)


def save_shallow_topic_user_emb():
    message_vocab = pickle.load(open(shallow_topic_message_path, 'rb'))
    user_message_data = pd.read_csv(user_message_relation_path, sep='\t')
    users = user_message_data['user']
    messages = user_message_data['message']
    numbers = len(users)
    shallow_topic_user_vocab = {}
    for i in range(numbers):
        user = users[i]
        print(i)
        try:
            message = messages[i].split(",")
        except:
            continue
        # print(message)
        len_message = len(message)
        user_vector = np.zeros([100], dtype=np.float32)
        if len_message > 0:
            for m in message:
                m_vector = message_vocab[m]
                for j in range(0, 100, 1):
                    user_vector[j] += m_vector[j]
            for k in range(0, 100, 1):
                user_vector[k] /= len_message
        print(user_vector)
        shallow_topic_user_vocab[user] = user_vector
    print(len(shallow_topic_user_vocab))
    with open(shallow_topic_user_path, 'wb') as g:
        pickle.dump(shallow_topic_user_vocab, g)

def save_word_user_emb():
    word_vocab = pickle.load(open(word_path, 'rb'))
    user_word_data = pd.read_csv(user_word_relation_path, sep='\t')
    users = user_word_data['user']
    words = user_word_data['word']
    numbers = len(users)
    word_user_vocab = {}
    for i in range(numbers):
        user = users[i]
        print(i)
        try:
            word = words[i].split(",")
        except:
            continue
        len_word = len(word)
        user_vector = np.zeros([100], dtype=np.float32)
        if len_word > 0:
            for w in word:
                w_vector = word_vocab[w]
                for j in range(0, 100, 1):
                    user_vector[j] += w_vector[j]
            for k in range(0, 100, 1):
                user_vector[k] /= len_word
        print(user_vector)
        word_user_vocab[user] = user_vector
    print(len(word_user_vocab))
    with open(word_user_path, 'wb') as g:
        pickle.dump(word_user_vocab, g)


def get_user_vocab():
    data = pd.read_csv(user_vocab_path, sep='\t')
    users = data['user']
    user_ids = data['user_id']
    user_vocab = {}
    for i in range(len(users)):
        user = users[i]
        id = user_ids[i]
        user_vocab[user] = id
        # print(user, id)
    return user_vocab


def create_training_content():
    user_vocab = get_user_vocab()
    hybird_topic_vocab = pickle.load(open(hybird_topic_path, 'rb'))
    hybird_word_vocab = pickle.load(open(hybird_word_path, 'rb'))
    shallow_topic_vocab = pickle.load(open(shallow_topic_user_path, 'rb'))
    label_path = "/media/iiip/Elements/数据集/user_profiling/weibo/label/gender.csv"
    data = pd.read_csv(label_path, sep='\t')
    labeled_user = data['user']
    labels = data['label']
    features = []
    y = []
    for i in range(len(labeled_user)):
        user = labeled_user[i]
        if user_vocab[user] not in hybird_topic_vocab:
            continue
        fe = np.concatenate([hybird_topic_vocab[user_vocab[user]], hybird_word_vocab[user_vocab[user]]])
        fe = np.concatenate([fe, shallow_topic_vocab[user_vocab[user]]])
        features.append(fe)
        # features.append(hybird_topic_vocab[user_vocab[user]])
        # features.append(shallow_topic_vocab[user_vocab[user]])
        y.append(int(labels[i]))
    y = np.array(y)
    features = np.array(features)
    print(len(features))
    print(features[0])
    return features, y

from sklearn.model_selection import StratifiedKFold
def train_test():
    X_features, y = create_training_content()
    print(np.shape(X_features))
    n_folds = 10
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    kf.get_n_splits(X_features, y)
    total_acc, total_pre, total_recall, total_macro_f1, total_micro_f1 = [], [], [], [], []
    for train_index, test_index in kf.split(X_features, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc, pre, recall, macro_f1, micro_f1 = classify(train_X=X_train,train_y=y_train, test_X=X_test, test_y=y_test)
        total_acc.append(acc)
        total_pre.append(pre)
        total_recall.append(recall)
        total_macro_f1.append(macro_f1)
        total_micro_f1.append(micro_f1)
        del X_train, X_test, y_train, y_test
    print("======================")
    print("avg acc:", np.mean(total_acc))
    print("avg pre:", np.mean(total_pre))
    print("avg recall:", np.mean(total_recall))
    print("avg macro_f1:", np.mean(total_macro_f1))
    print("avg micro_f1:", np.mean(total_micro_f1))
    print("======================")

# train_test()


# create_training_content()
# save_hybrid_topic_user_emb()
# save_hybird_word_user_emb()
# save_shallow_topic_emb()
# save_shallow_topic_user_emb()
# save_word_emb()
# save_word_user_emb()