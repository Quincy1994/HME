# coding=utf-8
import pandas as pd
from collections import defaultdict

data_path = "/media/iiip/Elements/数据集/user_profiling/weibo/weibo/user_weibo_seg_list.csv"
message_vocab_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/message_vocab.csv"
word_vocab_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/word_vocab.csv"
user_vocab_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_vocab.csv"
user_message_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_message_relation.csv"
message_word_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/message_word_relation.csv"
user_word_relation_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_word_relation.csv"

user_message_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_message_network.txt"
message_word_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/message_word_network.txt"
user_word_network_path = "/media/iiip/Elements/数据集/user_profiling/weibo/cache/HME/user_word_network.txt"



def message_to_index():
    data = pd.read_csv(data_path, sep='\t')
    contents = data['weibo']
    index = 1
    message_vocab = {}
    for content in contents:
       messages = str(content).split("|||")
       for message in messages:
           message = message.strip()
           if message not in message_vocab:
               message_vocab[message] = index
               index += 1
    data = []
    for message in message_vocab:
        if len(message) < 1:
            continue
        message_id = "M" + str(message_vocab[message])
        row = {}
        row['message'] = message
        row['message_id'] = message_id
        data.append(row)
    data = pd.DataFrame(data)
    data = data[["message", "message_id"]]
    data.fillna("", inplace=True)
    data.to_csv(message_vocab_path, index=False, sep='\t')
    print("message vocab has been done!")

def word_to_index():
    data = pd.read_csv(data_path, sep='\t')
    contents = data["weibo"]
    index = 1
    word_vocab = {}
    for content in contents:
        messages = str(content).split("|||")
        for message in messages:
            words = message.split(" ")
            for word in words:
                if word not in word_vocab:
                    word_vocab[word] = index
                    index += 1
    data = []
    for word in word_vocab:
        if len(word) < 1:
            continue
        word_id = "W" + str(word_vocab[word])
        row = {}
        row['word'] = word
        row['word_id'] = word_id
        data.append(row)
    data = pd.DataFrame(data)
    data = data[['word', 'word_id']]
    data.fillna("", inplace=True)
    data.to_csv(word_vocab_path, index=False, sep='\t')
    print("word vocab has been done!")

def user_to_index():
    data = pd.read_csv(data_path, sep='\t')
    users = data['user']
    index = 1
    user_vocab = {}
    for user in users:
        user_vocab[user] = index
        index += 1
    data = []
    for user in user_vocab:
        user_id = "U" + str(user_vocab[user])
        row = {}
        row["user"] = user
        row["user_id"] = user_id
        data.append(row)
    data = pd.DataFrame(data)
    data = data[['user', 'user_id']]
    data.fillna("", inplace=True)
    data.to_csv(user_vocab_path, index=False, sep='\t')
    print("user vocab has been done!")


def load_message_vocab():
    message_data = pd.read_csv(message_vocab_path, sep='\t')
    messages = message_data['message']
    ids = message_data['message_id']
    message_vocab = {}
    for i in range(len(ids)):
        message_vocab[messages[i]] = ids[i]
    return message_vocab

def load_user_vocab():
    user_data = pd.read_csv(user_vocab_path, sep='\t')
    users = user_data['user']
    ids = user_data['user_id']
    user_vocab = {}
    for i in range(len(ids)):
        user_vocab[users[i]] = ids[i]
    return user_vocab

def load_word_vocab():
    word_data = pd.read_csv(word_vocab_path, sep='\t')
    words = word_data['word']
    ids = word_data['word_id']
    word_vocab = {}
    for i in range(len(ids)):
        word_vocab[words[i]] = ids[i]
    return word_vocab

def save_network_pair(pairs, pairs_path):
    f = open(pairs_path, 'w')
    for pair in pairs:
        line = str(pair[0]) + " " + str(pair[1]) + " " + str(pairs[pair])
        f.write(line + "\n")
    f.close()


def construct_network():
    message_vocab = load_message_vocab()
    user_vocab = load_user_vocab()
    word_vocab = load_word_vocab()
    data = pd.read_csv(data_path, sep='\t')
    users = data['user']
    contents = data['weibo']
    number = len(users)

    user_message_data = []
    message_word_data = []
    user_word_data = []

    user_message_pairs = defaultdict(int)
    message_word_pairs = defaultdict(int)
    user_word_pairs = defaultdict(int)

    for i in range(number):
        print(i)
        user = users[i]
        user_id = user_vocab[user]
        message_list = []
        user_word_list = []
        content = contents[i]
        messages = str(content).split("|||")
        for message in messages:
            if message in message_vocab:
                message_id = message_vocab[message]
                message_list.append(message_id)
                user_message_pair = (user_id, message_id)
                user_message_pairs[user_message_pair] += 1
                words = message.split(" ")

                word_list = []
                for word in words:
                    if word in word_vocab:
                        word_id = word_vocab[word]
                        user_word_list.append(word_id)
                        word_list.append(word_id)
                        message_word_pair = (message_id, word_id)
                        message_word_pairs[message_word_pair] += 1
                        user_word_pair = (user_id, word_id)
                        user_word_pairs[user_word_pair] += 1
                mw_row = {}
                mw_row['message'] = message_id
                mw_row['word'] = ",".join(word_list)
                message_word_data.append(mw_row)

        um_row = {}
        um_row['user'] = user_id
        um_row['message'] = ",".join(message_list)
        user_message_data.append(um_row)

        uw_row = {}
        uw_row['user'] = user_id
        uw_row['word'] = ",".join(user_word_list)
        user_word_data.append(uw_row)

    save_network_pair(user_message_pairs, user_message_network_path)
    save_network_pair(message_word_pairs, message_word_network_path)
    save_network_pair(user_word_pairs, user_word_network_path)

    user_word_data = pd.DataFrame(user_word_data)
    user_word_data = user_word_data[['user', 'word']]
    user_word_data.fillna("", inplace=True)
    user_word_data.to_csv(user_word_relation_path, index=False, sep='\t')
    print("user word relation has been done!")

    user_message_data = pd.DataFrame(user_message_data)
    user_message_data = user_message_data[['user', 'message']]
    user_message_data.fillna("", inplace=True)
    user_message_data.to_csv(user_message_relation_path, index=False, sep='\t')
    print("user message relation has been done!")

    message_word_data = pd.DataFrame(message_word_data)
    message_word_data = message_word_data[['message', 'word']]
    message_word_data.fillna("", inplace=True)
    message_word_data.to_csv(message_word_relation_path, index=False, sep='\t')
    print("message word relation has been done!")



# message_to_index()
# word_to_index()
# user_to_index()
construct_network()