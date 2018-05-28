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
 
# def get_fb_data_data(download=False):
#     PATH='data.msgpack'
#     if download:
#         wget.download('https://www.dropbox.com/s/r33ljlagys0wscb/data.msgpack?dl=1', out=PATH)
#     with open(PATH, 'rb') as f:
#         data = msgpack.load(f, encoding='utf8')
#     return data
 
# def get_fb_data_meta(download=False):
#     PATH='meta.msgpack'
#     if download:
#         wget.download('https://www.dropbox.com/s/83txkgiqmdlv1m3/meta.msgpack?dl=1', out=PATH)
#     with open(PATH, 'rb') as f:
#         data = msgpack.load(f, encoding='utf8')
#     return data
 

# def F(beg_p, end_p, beg, end):
#     TP = min(end, end_p) - max(beg_p, beg)
#     if TP < 0:
#         TP = 0
#     FP = max(end, end_p) - min(beg_p, beg) - TP
#     if FP < 0:
#         FP = 0
#     if FP == 0 and TP == 0:
#         FP = 10
#     return ((2 * TP) / (2 * TP + FP))
 
# def F_score(beg_p_v, end_p_v, beg_v, end_v, size):
#     res = 0
#     for i in range(size):
#         beg_p = np.argmax(beg_p_v[i])
#         end_p = np.argmax(end_p_v[i])
#         beg = beg_v[i]
#         end = end_v[i]
       
#         res += F(beg_p, end_p, beg, end)
#     return res/size
 
# def F_dif_score(beg_p_v, end_p_v, beg_v, end_v, size, context):
#     res = 0
#     for i in range(size):
#         maxim, beg_p, end_p = 0, 0, 0
#         for j in range(context):
#             for k in range(j, min(j + 15, context)):
#                 if maxim < beg_p_v[i][j] * end_p_v[i][k]:
#                     maxim, beg_p, end_p= c[j][k], j, k
#         beg = beg_v[i]
#         end = end_v[i]
#         res += F(beg_p, end_p, beg, end)
#     return res/size

 
# def get_demo_emb(context, question, meta, word2index):
#     pivot = BATCH_SIZE
       
#     context_embedded = np.zeros((pivot, CONTEXT_MAX_SIZE, EMBEDDING_SIZE))
#     context_len = np.zeros((pivot), dtype=np.int32)
#     features = np.zeros((pivot, CONTEXT_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)
   
#     question_embedded = np.zeros((pivot, QUESTION_MAX_SIZE, EMBEDDING_SIZE))
#     masked_question = np.zeros((pivot), dtype=np.int32)
 
#     count = 0
#     # на случай если такого слова нет, надо ????
#     tokenized_context = tokenize(context)
#     print("tokenized context")
#     print(tokenized_context)
   
#     nums_context = [word2index.get(word, 1) for word in tokenized_context]
 
#     context_len[0] = len(nums_context)
   
#     for j in range(len(nums_context)):
       
#         context_embedded[0][j] = meta['embedding'][nums_context[j]]
       
#         features[0][j][:4] = [False, False, False, 0.0]
       
#         ## тут потом можно взять энтитис
#         #features[0][j][len(data[shuffled_indexes[i]][2][j]) + data[shuffled_indexes[i]][4][j]] = 1
#         #features[0][j][len(data[shuffled_indexes[i]][2][j]) + ent_num + data[shuffled_indexes[i]][3][j]] = 1
       
#     for j in range(len(nums_context), CONTEXT_MAX_SIZE):
#         context_embedded[0][j] = meta['embedding'][0]
 
#     nums_question = [word2index.get(word, 1) for word in tokenize(question)]
#     masked_question[0] = len(nums_question)
 
#     for j in range(len(nums_question)):
#         question_embedded[0][j] = meta['embedding'][nums_question[j]]
#     for j in range(len(nums_question), QUESTION_MAX_SIZE):
#         question_embedded[0][j] = meta['embedding'][0]
       
#     return context_embedded, context_len, features, question_embedded, masked_question
 


def main():
 
    meta = get_fb_data_meta()
 
    word2index = {w: i for i, w in enumerate(meta['vocab'])}
    tag2index = {w: i for i, w in enumerate(meta['vocab_tag'])}
    ent2index = {w: i for i, w in enumerate(meta['vocab_ent'])}
 
    tf.reset_default_graph()
    sess_r=tf.Session()
    saver = tf.train.import_meta_graph(USE_MODEL_PATH)
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
 
    print("ДЛЯ ВЫХОДА нажмите q")
    stop_flag=False
    while True:
        id_ = 0
        try:
            while True:
                context = input('Контекст для вопроса ')
                if (context == "q"):
                    stop_flag=True
                    print("FINISHING")
                    break
                   
                if context.strip():
                    break                
                   
            if stop_flag:
                break
               
            while True:
                question = input('Вопрос ')
                if question.strip():
                    break
        except EOFError:
            break
 
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
        current_cont_len = len(tock_context)
       
        print(current_cont_len)
        print(b)
        print(e)
       
        if (e >= current_cont_len):
            print("Something went wrong with end - optimizing answer")
            e = current_cont_len - 1
        if (b >= current_cont_len):
            print("Something went wrong with begin - optimizing answer")
            b = current_cont_len - 1
        print("**********Ваш ответ*********")
        print(normalize(tock_context[b:e + 1]))
 
 
if __name__ == '__main__':
    main()