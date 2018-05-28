import msgpack
import tensorflow as tf
import numpy as np
from constants import*
from prepare import*

questions = tf.placeholder(dtype=tf.float32, name="questions", shape=(None, QUESTION_MAX_SIZE, EMBEDDING_SIZE))
answer_starts = tf.placeholder(dtype=tf.int32, name="answer_starts", shape=(None))
answer_ends = tf.placeholder(dtype=tf.int32, name="answer_ends", shape=(None))
mask_q = tf.placeholder(dtype=tf.int32, name="mask_q", shape=(None))

embedded_inputs_nof = tf.placeholder(dtype=tf.float32, name="embedded_inputs_nof", shape=(None, CONTEXT_MAX_SIZE, EMBEDDING_SIZE))

emb_features = tf.placeholder(dtype=tf.float32, name="emb_features", shape=(None, CONTEXT_MAX_SIZE, FEATURES_SIZE))
seq_len = tf.placeholder(dtype=tf.int32, name="seq_len", shape=(None))

keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob", shape=())
if_train = tf.placeholder(dtype=tf.bool, name="if_train" )
#==================
b_s = tf.shape(seq_len)[0]
embedded_inputs = tf.concat([embedded_inputs_nof, emb_features], axis = -1)
#==================
lstm_bw_cell_q = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
lstm_bw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_q, keep_prob)
lstm_fw_cell_q = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
lstm_fw_cell_q = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_q, keep_prob)
question_outputs, question_state = tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell_q,
    lstm_bw_cell_q,
    inputs=questions,
    sequence_length=mask_q,
    dtype=tf.float32
    
)
lstm_bw_cell_q2 = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
lstm_bw_cell_q2 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_q2, keep_prob)
lstm_fw_cell_q2 = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_HIDDEN_SIZE)
lstm_fw_cell_q2 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_q2, keep_prob)
question_input = tf.concat([question_outputs[0], question_outputs[1]], -1)

question_outputs, question_state = tf.nn.bidirectional_dynamic_rnn(
    lstm_fw_cell_q2,
    lstm_bw_cell_q2,
    inputs=question_input,
    sequence_length=mask_q,
    dtype=tf.float32,
    scope="BiLSTM"
    
)
#==================


question_output = tf.concat(question_outputs, -1)
question_attention = tf.layers.dense(
        inputs=question_output,
        units=1,
        use_bias=True
)
question_mask = tf.sequence_mask(
        mask_q,
        maxlen=QUESTION_MAX_SIZE,
        dtype=tf.float32
)
question_attention = tf.reshape(question_attention, (b_s, QUESTION_MAX_SIZE))
question_attention = tf.multiply(tf.nn.softmax(question_attention), question_mask)
question_attention = tf.reshape(question_attention, (b_s, QUESTION_MAX_SIZE, 1))

#[.][.][.][.]   [.]
#[.][.][.][.] * [.]
#[.][.][.][.]   [.]//weights

question_output = tf.matmul(
	tf.transpose(question_output, perm=[0, 2, 1]), question_attention
	)
q_state = tf.contrib.rnn.LSTMStateTuple(question_output, question_output)
#==================
question_output = tf.reshape(
	question_output, (b_s, 2 * LSTM_CELL_HIDDEN_SIZE)
	)
#!!!!!!!question_output = tf.layers.batch_normalization(question_output, training = if_train)
#==================
lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, keep_prob)


lstm_fw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
lstm_fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_1, keep_prob)
lstm_bw_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(2 * LSTM_CELL_HIDDEN_SIZE)
lstm_bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_1, keep_prob)

context_outputs, context_state = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=lstm_fw_cell, 
    initial_state_fw = q_state,
    cell_bw=lstm_bw_cell,
    initial_state_bw = q_state,
    sequence_length=seq_len,
    inputs=embedded_inputs,
    dtype=tf.float32,
    scope="BiLSTM0"
)
print(context_outputs[0])
context_input = tf.concat([context_outputs[0], context_outputs[1]], -1)
#!!!!!!!context_input = tf.layers.batch_normalization(context_input, training = if_train)
context_outputs1, context_state1 = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=lstm_fw_cell_1,
    cell_bw=lstm_bw_cell_1,
    initial_state_fw = q_state,
    initial_state_bw = q_state,
    inputs=context_input,
    dtype=tf.float32,
    scope="BiLSTM1"
)
print(context_outputs1)
context_outputs = tf.concat(context_outputs1, -1)
#!!!!!!!context_outputs = tf.layers.batch_normalization(context_outputs, training = if_train)
print(1)
#==================
dense_begin = tf.layers.dense(
        inputs=context_outputs,
        units=2*LSTM_CELL_HIDDEN_SIZE,
        use_bias=True,
    
)

#dense_begin = tf.layers.batch_normalization(dense_begin)
dense_end = tf.layers.dense(
        inputs=context_outputs,
        units=2*LSTM_CELL_HIDDEN_SIZE,
        use_bias=True
)
#dense_end = tf.layers.batch_normalization(dense_end)
dense_begin = tf.matmul(dense_begin,
				tf.reshape(question_output,
				(b_s, 2*LSTM_CELL_HIDDEN_SIZE, 1)),
				name = "dense_begin"
				)
dense_end = tf.matmul(dense_end,
				tf.reshape(question_output,
				(b_s, 2*LSTM_CELL_HIDDEN_SIZE, 1)),
				name = "dense_end"
				)
loss_beg = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer_starts,
        logits=tf.reshape(dense_begin, (b_s, CONTEXT_MAX_SIZE)),
        name="softmax_begin"
    )
)
loss_end = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer_ends,
        logits=tf.reshape(dense_end, (b_s, CONTEXT_MAX_SIZE)),
        name="softmax_end"
    )
)
print(1)
loss = loss_beg + loss_end
optimizer = tf.train.RMSPropOptimizer(0.001)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 1000)
train_step = optimizer.apply_gradients(zip(gradients, variables))
#@title Default title text
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

saver = tf.train.Saver()
for i in (range(TRAINING_EPOCHS)):
    (emb_t, len_emb_t, features_t, q_t, start_t, end_t, mask_q_t) = get_new_emb(BATCH_SIZE, train)
    loss_val, _, b, e= sess.run([loss,
                                         train_step, 
                                         dense_begin,
                                         dense_end
                                        ], 
                                 feed_dict={
                                     embedded_inputs_nof: emb_t,
                                     answer_starts: start_t,
                                     answer_ends: end_t,
                                     emb_features:features_t,
                                     questions: q_t,
                                     if_train: True,
                                     keep_prob: 0.7,
                                     mask_q: mask_q_t,
                                     seq_len: len_emb_t}
                             )
    
    if i % 100 == 0:
        (emb_d, len_emb_d, features_d, q_d, start_d, end_d, mask_q_d) = get_new_emb(BATCH_SIZE, data['dev'], False)
    if i % 10 == 0:
        F2 = F_score(b, e, start_t, end_t, BATCH_SIZE)
        print(str(i) + ": " + str(loss_val) + "\t\t" +str(F2))
        
        b_d, e_d = sess.run(
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
        F2_t = F_score(b_d, e_d, start_d, end_d, BATCH_SIZE)
        print("test:\t\t\t" + str(F2_t))

    if i % (TRAINING_EPOCHS // 2) == 0:
        saver.save(sess, './datas/model_2/my_test_model_' + str(i))
saver.save(sess, './datas/model_2/res/actual')








