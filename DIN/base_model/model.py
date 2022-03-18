import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):
  # cate_list, self.hist_i -> hc
  # item_emb_w, self.hist_i, cate_emb_w, hc -> h_emb

  # cate_list, self.i -> ic
  # item_emb_w, self.i, cate_emb_w, ic -> i_emb

  # cate_list, self.j -> jc
  # item_emb_w, self.j, cate_emb_w, jc -> j_emb

  # self.sl, h_emb -> u_emb -> u_emb_all

  # u_emb, i_emb -> d_layer_3_i
  # u_emb, j_emb -> d_layer_3_j

  # d_layer_3_i, i_b -> self.logits

  # item_emb_w, cate_list, cate_emb_w -> all_emb
  # u_emb_all, all_emb -> din_all -> d_layer_3_all
  # item_b, d_layer_3_all -> self.logits_all

  def __init__(self, user_count, item_count, cate_count, cate_list):
    self.u = tf.placeholder(tf.int32, [None,]) # [B]
    self.i = tf.placeholder(tf.int32, [None,]) # [B]
    self.j = tf.placeholder(tf.int32, [None,]) # [B]
    self.y = tf.placeholder(tf.float32, [None,]) # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
    self.sl = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units]) # [U, H]
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) # [I, H/2]
    item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) # [I]
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])  # [C, H/2]
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64) # [I]

    u_emb = tf.nn.embedding_lookup(user_emb_w, self.u) # [B, H]

    ic = tf.gather(cate_list, self.i) # [B]
    i_emb = tf.concat(values = [ # [B, H]
        tf.nn.embedding_lookup(item_emb_w, self.i), # [B, H/2]
        tf.nn.embedding_lookup(cate_emb_w, ic), # [B, H/2]
        ], axis=1)
    i_b = tf.gather(item_b, self.i) # [B]

    jc = tf.gather(cate_list, self.j) # [B]
    j_emb = tf.concat([ # [B, H]
        tf.nn.embedding_lookup(item_emb_w, self.j), # [B, H/2]
        tf.nn.embedding_lookup(cate_emb_w, jc), # [B, H/2]
        ], axis=1)
    j_b = tf.gather(item_b, self.j) # [B]

    hc = tf.gather(cate_list, self.hist_i) # [B, T]
    h_emb = tf.concat([ # [B, T, H]
        tf.nn.embedding_lookup(item_emb_w, self.hist_i), # [B, T, H/2]
        tf.nn.embedding_lookup(cate_emb_w, hc), # [B, T, H/2]
        ], axis=2)

    #-- sum pooling of user behavior -------
    mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
    mask = tf.expand_dims(mask, -1) # [B, T, 1]
    mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]

    hist = h_emb * mask
    hist = tf.reduce_sum(hist, 1) # [B, H]
    hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl,1), [1,tf.shape(h_emb)[2]]), tf.float32)) # [B, H]
    #-- sum end ---------
    
    hist = tf.layers.batch_normalization(inputs = hist)
    hist = tf.reshape(hist, [-1, hidden_units]) # [B, H]
    hist = tf.layers.dense(hist, hidden_units) # [B, H]
    u_emb = hist

    #-- fcn begin -------
    din_i = tf.concat([u_emb, i_emb], axis=-1) # [B, 2H]
    din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
    d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1') # [B, 80]
    d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2') # [B, 40]
    d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3') # [B, 1]
    d_layer_3_i = tf.reshape(d_layer_3_i, [-1]) # [B]

    din_j = tf.concat([u_emb, j_emb], axis=-1) # [B, 2H]
    din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
    d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
    d_layer_3_j = tf.reshape(d_layer_3_j, [-1]) # [B]

    x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
    self.logits = i_b + d_layer_3_i


    u_emb_all = tf.expand_dims(u_emb, 1) # [B, 1, H]
    u_emb_all = tf.tile(u_emb_all, [1, item_count, 1]) # [B, I, H]
    
    # logits for all item:
    all_emb = tf.concat([ # [I, H]
        item_emb_w, # [I, H/2]
        tf.nn.embedding_lookup(cate_emb_w, cate_list) # [I, H/2]
        ], axis=1)
    all_emb = tf.expand_dims(all_emb, 0) # [1, I, H]
    all_emb = tf.tile(all_emb, [512, 1, 1]) # [B, I, H]

    din_all = tf.concat([u_emb_all, all_emb], axis=-1) # [B, I, 2H]
    din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True) # [B, I, 1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count]) # [B, I]
    self.logits_all = tf.sigmoid(item_b + d_layer_3_all) # [B, I]
    #-- fcn end -------

    self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
    self.score_i = tf.sigmoid(i_b + d_layer_3_i) # [B]
    self.score_j = tf.sigmoid(j_b + d_layer_3_j) # [B]
    self.score_i = tf.reshape(self.score_i, [-1, 1]) # [B, 1]
    self.score_j = tf.reshape(self.score_j, [-1, 1]) # [B, 1]
    self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1) # [B, 2]

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.y)
        )

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        })
    return loss


  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.p_and_n], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.j: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        })
    return u_auc, socre_p_and_n


  def test(self, sess, uid, hist_i, sl):
    return sess.run(self.logits_all, feed_dict={
        self.u: uid,
        self.hist_i: hist_i,
        self.sl: sl,
        })


  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)


  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)


def extract_axis_1(data, ind):
  batch_range = tf.range(tf.shape(data)[0])
  indices = tf.stack([batch_range, ind], axis=1)
  res = tf.gather_nd(data, indices)
  return res

