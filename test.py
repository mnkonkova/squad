import msgpack
import tensorflow as tf
import numpy as np

def get_test_emb(size, data, l, tr=False, b = 0):
    m = len(data)
    l = l[b:min(size + b, m)]
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

sess_r=tf.Session() 
saver = tf.train.import_meta_graph('my_test_model.meta')
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

i = 0
F_d = 0
m = len(data['dev'])
l_0 = np.arange(0, m, 1)
while(i < m):
    (emb_d, len_emb_d, features_d, q_d, start_d, end_d, mask_q_d) = get_test_emb(BATCH_SIZE, data['dev'], l_0, False, b = i)
    b_d, e_d = sess_r.run(
                [dense_begin,
                 dense_end],
                feed_dict = {
                    embedded_inputs_nof: emb_d,
                    answer_starts: start_d, 
                    answer_ends: end_d,
                    emb_features:features_d,
                    questions: q_d,
                    if_train: False, 
                    keep_prob: 1.0,
                    mask_q: mask_q_d,
                    seq_len: len_emb_d
                }
            )
    print("done", i, "out of" , m)
    F_d += F_score(b_d, e_d, start_d, end_d, BATCH_SIZE) * BATCH_SIZE
    i += BATCH_SIZE
i  -= BATCH_SIZE
print('Test:\t', F_d / i) 