import msgpack
import json
import re
import numpy as np
import collections
import random
import gensim
import tensorflow as tf
 
from nltk.tokenize import word_tokenize
from tqdm import tqdm_notebook as tqdm
from nltk.tokenize import RegexpTokenizer

from constants import *
from prepare import *

def tokenize(text, lower=True):
    tokenizer = RegexpTokenizer('\w+|\$[\d]+|\S+')
    return [token.lower() for token in tokenizer.tokenize(text) if token.isalnum()]

def get_demo_emb(context, question, meta, word2index):
       
    emb = np.zeros((BATCH_SIZE, CONTEXT_MAX_SIZE, EMBEDDING_SIZE))
    len_emb = np.zeros((BATCH_SIZE), dtype=np.int32)

    start = np.zeros((BATCH_SIZE))
    end = np.zeros((BATCH_SIZE))
    emb_q = np.zeros((BATCH_SIZE, QUESTION_MAX_SIZE, EMBEDDING_SIZE))
    mask_q = np.zeros((BATCH_SIZE), dtype=np.int32)
    features = np.zeros((BATCH_SIZE, CONTEXT_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)

    tokenized_context = tokenize(context)
    nums_context = [word2index.get(word, 1) for word in tokenized_context]
    
    len_emb[0] = len(nums_context)
    for j in range(len(nums_context)):
       
        emb[0][j] = meta['embedding'][nums_context[j]]
       
        features[0][j][:4] = [False, False, False, 0.0]

    for j in range(len(nums_context), CONTEXT_MAX_SIZE):
        emb[0][j] = meta['embedding'][0]
 
    nums_question = [word2index.get(word, 1) for word in tokenize(question)]
    mask_q[0] = len(nums_question)
 
    for j in range(len(nums_question)):
        emb_q[0][j] = meta['embedding'][nums_question[j]]
    for j in range(len(nums_question), QUESTION_MAX_SIZE):
        emb_q[0][j] = meta['embedding'][0]
       
    return emb, len_emb, features, emb_q, mask_q
 


word2index = {w: i for i, w in enumerate(meta['vocab'])}
tag2index = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2index = {w: i for i, w in enumerate(meta['vocab_ent'])}

tf.reset_default_graph()
sess_r=tf.Session()
saver = tf.train.import_meta_graph('my_test_model_1.meta')
saver.restore(sess_r,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
embedded_inputs_nof = graph.get_tensor_by_name("embedded_inputs_nof:0")
answer_starts = graph.get_tensor_by_name("answer_starts:0")
answer_ends = graph.get_tensor_by_name("answer_ends:0")
emb_features = graph.get_tensor_by_name("emb_features:0")
questions = graph.get_tensor_by_name("questions:0")
if_train = graph.get_tensor_by_name("if_train:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
mask_q = graph.get_tensor_by_name("mask_q:0")
seq_len =  graph.get_tensor_by_name("seq_len:0")

dense_begin = graph.get_tensor_by_name("dense_begin:0")
dense_end = graph.get_tensor_by_name("dense_end:0")
print("Context:")
context = str(input())
print("Question:")
question = str(input())

(cont_emb, cont_len, feats, quest_emb, quest_mask) = get_demo_emb(context, question, meta, word2index)

b_d, e_d = sess_r.run(
        [dense_begin,
         dense_end],
        feed_dict = {
            embedded_inputs_nof: cont_emb,
            emb_features: feats,
            questions: quest_emb,
            if_train: False,
            keep_prob: 1.0,
            mask_q: quest_mask,
            seq_len: cont_len
        }
    )

b = np.argmax(b_d[0].reshape((-1, )))
e = np.argmax(e_d[0].reshape((-1, )))

tock_context = tokenize(context)
c_l = len(tock_context)
if (e >= c_l):
    e = c_l - 1
if (b >= c_l):
    b = c_l - 1
print("Answer:")
print(" ".join((tock_context[b:e])))