import msgpack
import tensorflow as tf
import numpy as np
from constants import*


def F(beg_p, end_p, beg, end):
    TP = min(end, end_p) - max(beg_p, beg)
    if TP < 0:
        TP = 0
    FP = max(end, end_p) - min(beg_p, beg) - TP
    if FP < 0:
        FP = 0
    if FP == 0 and TP == 0:
        FP = 10
    return ((2 * TP) / (2 * TP + FP))
def F_score(beg_p_v, end_p_v, beg_v, end_v, size):
    res = 0
    for i in range(size):
        beg_p = np.argmax(beg_p_v[i])
        end_p = np.argmax(end_p_v[i])
        beg = beg_v[i]
        end = end_v[i]
        
        res += F(beg_p, end_p, beg, end)
    return res/size     
def F_dif_score(beg_p_v, end_p_v, beg_v, end_v, size, context):
    res = 0
    for i in range(size):
        maxim, beg_p, end_p = 0, 0, 0
        for j in range(context):
            for k in range(j, min(j + 15, context)):
                if maxim < beg_p_v[i][j] * end_p_v[i][k]:
                    maxim, beg_p, end_p= c[j][k], j, k
        beg = beg_v[i]
        end = end_v[i]
        res += F(beg_p, end_p, beg, end)
    return res/size   
def get_new_emb(size, data, tr=True):
    m = len(data)
    l = np.arange(0, m, 1)
    np.random.shuffle(l)
    l = l[:size]
    emb = np.zeros((size, CONTEXT_MAX_SIZE, EMBEDDING_SIZE))
    len_emb = np.zeros((size), dtype=np.int32)
    
    start = np.zeros((size))
    end = np.zeros((size))
    emb_q = np.zeros((size, QUESTION_MAX_SIZE, EMBEDDING_SIZE))
    mask_q = np.zeros((size), dtype=np.int32)
    features = np.zeros((size, CONTEXT_MAX_SIZE, FEATURES_SIZE), dtype=np.float64)
    
    for i in range(size):
        count = 0
        par = data[l[i]][1]
        
        len_emb[i] = len(par)
        for j in range(len(par)):
            emb[i][j] = meta['embedding'][par[j]]
            features[i][j][:len(data[l[i]][2][j])] = data[l[i]][2][j]
            features[i][j][len(data[l[i]][2][j]) + data[l[i]][4][j]] = 1
            features[i][j][len(data[l[i]][2][j]) + ent_num + data[l[i]][3][j]] = 1
        for j in range(len(par), CONTEXT_MAX_SIZE):
            emb[i][j] = meta['embedding'][0]
        
        
        quest = data[l[i]][5]
            
        
        mask_q[i] = len(quest)
        
        if tr:
            start[i] = data[l[i]][8]
            end[i] = data[l[i]][9]
        else:
            start[i] = answers_be[l[i]][0]
            end[i] = answers_be[l[i]][1]
            
        for j in range(len(quest)):
            emb_q[i][j] = meta['embedding'][quest[j]]
        for j in range(len(quest), QUESTION_MAX_SIZE):
            emb_q[i][j] = meta['embedding'][0]
    return emb, len_emb, features, emb_q, start, end, mask_q
def find_beg(par, ans, tokens):
    ind = par.find(ans)
    beg = len(tokens)
    for i in range(len(tokens)):
        if tokens[i][0] == ind:
            beg = i
            break
    return beg
def find_end(beg, par, ans, tokens):
    ind = par.find(ans)
    prob_end = ind + len(ans)
    end = len(tokens)
    for i in range(beg, len(tokens)):
        if tokens[i][1] == prob_end:
            end = i
            break
    return end + 1

with open('data.msgpack', 'rb') as f:
    data = msgpack.load(f, encoding='utf8')
train = data['train']
with open('meta.msgpack', 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')

answers_be = np.zeros((len(data['dev']), 2))
for i in range(len(data['dev'])):
    beg = find_beg(data['dev'][i][6], data['dev'][i][8][0],data['dev'][i][7])
    answers_be[i][0] = beg
    answers_be[i][1] = find_end(beg, data['dev'][i][6], data['dev'][i][8][0],data['dev'][i][7])
