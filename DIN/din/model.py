import tensorflow as tf

from Dice import dice

class Model(object):
  # item_emb_w, self.user_items, cate_emb_w, users_items_cates -> user_items_cates_emb

  # item_emb_w, self.pos_items, cate_emb_w, pos_cates -> pos_items_cates_emb
  # pos_items_cates_emb, user_items_cates_emb -> pos_user_items_cates_attention
  # pos_user_items_cates_attention, pos_items_cates_emb -> pos_din_output

  # item_emb_w, self.neg_items, cate_emb_w, neg_cates -> neg_items_cates_emb
  # neg_items_cates_emb, user_items_cates_emb -> neg_user_items_cates_attention
  # neg_user_items_cates_attention, neg_items_cates_emb -> neg_din_output

  # item_emb_w, cate_emb_w, cate_list -> items_cates_emb
  # items_cates_emb, predict_ads_num -> ads_cates_emb
  # ads_cates_emb, user_items_cates_emb -> user_ads_cates_attention
  # user_ads_cates_attention, ads_cate_emb -> ads_din_output

  # pos_din_output, pos_items_emb_b -> self.logits, pos_score
  # neg_din_output, neg_items_emb_b -> self.logits, neg_score
  # ads_din_output, predict_ads_num, item_emb_b -> self.ads_logits

  def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
    self.users = tf.placeholder(tf.int32, [None,]) # [B]
    self.pos_items = tf.placeholder(tf.int32, [None,]) # [B]
    self.neg_items = tf.placeholder(tf.int32, [None,]) # [B]
    self.label = tf.placeholder(tf.float32, [None,]) # [B]
    self.user_items = tf.placeholder(tf.int32, [None, None]) # [B, T] user-item seq, T is the length of seq
    self.seq_len = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, [])

    hidden_units = 128

    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) # [I, H//2]
    item_emb_b = tf.get_variable("item_emb_b", [item_count], initializer=tf.constant_initializer(0.0)) # [I]
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2]) # [C, H//2]
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64) # I

    pos_cates = tf.gather(cate_list, self.pos_items) # [B]
    pos_items_cates_emb = tf.concat([ # [B, H]
                                    tf.nn.embedding_lookup(item_emb_w, self.pos_items), # [B, H/2]
                                    tf.nn.embedding_lookup(cate_emb_w, pos_cates), # [B, H/2]
                                  ], axis=1)
    pos_items_emb_b = tf.gather(item_emb_b, self.pos_items) # [B]

    neg_cates = tf.gather(cate_list, self.neg_items) # [B]
    neg_items_cates_emb = tf.concat([ # [B, H]
                                    tf.nn.embedding_lookup(item_emb_w, self.neg_items), # [B, H/2]
                                    tf.nn.embedding_lookup(cate_emb_w, neg_cates), # [B, H/2]
                                  ], axis=1)
    neg_items_emb_b = tf.gather(item_emb_b, self.neg_items) # [B]

    # get user behavior cates
    user_items_cates = tf.gather(cate_list, self.user_items) # [B, T]
    # get user behavior embeddings
    user_items_cates_emb = tf.concat([ # [B, T, H]
                                    tf.nn.embedding_lookup(item_emb_w, self.user_items), # [B, T, H/2]
                                    tf.nn.embedding_lookup(cate_emb_w, user_items_cates), # [B, T, H/2]
                                   ], axis=2)

    pos_user_items_cates_attention = attention(pos_items_cates_emb, user_items_cates_emb, self.seq_len) # [B, 1, H]
    pos_user_items_cates_attention = tf.layers.batch_normalization(inputs = pos_user_items_cates_attention)
    pos_user_items_cates_attention = tf.reshape(pos_user_items_cates_attention, [-1, hidden_units], name='hist_bn') # [B, H]
    pos_user_items_cates_attention = tf.layers.dense(pos_user_items_cates_attention, hidden_units, name='hist_fcn') # [B, H]

    neg_user_items_cates_attention = attention(neg_items_cates_emb, user_items_cates_emb, self.seq_len)
    # neg_user_items_cates_attention = tf.layers.batch_normalization(inputs = neg_user_items_cates_attention)
    neg_user_items_cates_attention = tf.layers.batch_normalization(inputs = neg_user_items_cates_attention, reuse = True)
    neg_user_items_cates_attention = tf.reshape(neg_user_items_cates_attention, [-1, hidden_units], name='hist_bn') # [B, H]
    neg_user_items_cates_attention = tf.layers.dense(neg_user_items_cates_attention, hidden_units, name='hist_fcn', reuse=True)

    # -- fcn begin -------
    pos_din = tf.concat([pos_user_items_cates_attention, pos_items_cates_emb, 
                          pos_user_items_cates_attention * pos_items_cates_emb], axis=-1) # [B, 3H]
    pos_din = tf.layers.batch_normalization(inputs=pos_din, name='b1')
 
    pos_din_layer_1 = tf.layers.dense(pos_din, 80, activation=tf.nn.sigmoid, name='f1') # [B, 80]
    #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
    # pos_din_layer_1 = tf.layers.dense(pos_din, 80, activation=None, name='f1')
    # pos_din_layer_1 = dice(pos_din_layer_1, name='dice_1_i')
    pos_din_layer_2 = tf.layers.dense(pos_din_layer_1, 40, activation=tf.nn.sigmoid, name='f2') # [B, 40]
    # pos_din_layer_2 = tf.layers.dense(pos_din_layer_1, 40, activation=None, name='f2')
    # pos_din_layer_2 = dice(pos_din_layer_2, name='dice_2_i')
    pos_din_output = tf.layers.dense(pos_din_layer_2, 1, activation=None, name='f3') # [B, 1]

    neg_din = tf.concat([neg_user_items_cates_attention, neg_items_cates_emb, 
                          neg_user_items_cates_attention * neg_items_cates_emb], axis=-1) # [B, 3H]
    neg_din = tf.layers.batch_normalization(inputs=neg_din, name='b1', reuse=True)

    neg_din_layer_1 = tf.layers.dense(neg_din, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    # neg_d_layer_1 = tf.layers.dense(neg_din, 80, activation=None, name='f1', reuse=True)
    # neg_d_layer_1 = dice(neg_d_layer_1, name='dice_1_j')
    neg_din_layer_2 = tf.layers.dense(neg_din_layer_1, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    # neg_d_layer_2 = tf.layers.dense(neg_d_layer_1, 40, activation=None, name='f2', reuse=True)
    # neg_d_layer_2 = dice(neg_d_layer_2, name='dice_2_j')
    neg_din_output = tf.layers.dense(neg_din_layer_2, 1, activation=None, name='f3', reuse=True) # [B, 1]

    pos_din_output = tf.reshape(pos_din_output, [-1]) # [B]
    neg_din_output = tf.reshape(neg_din_output, [-1]) # [B]

    pos_neg_diff = pos_items_emb_b + pos_din_output - (neg_items_emb_b + neg_din_output) # [B]
    self.logits = pos_items_emb_b + pos_din_output # [B]

    # prediciton for selected items
    # logits for selected item:
    items_cates_emb = tf.concat([ # [I, H]
                                item_emb_w,
                                tf.nn.embedding_lookup(cate_emb_w, cate_list)
                              ], axis=1)
    ads_cates_emb = items_cates_emb[:predict_ads_num, :] # [Ads, H]
    ads_cates_emb = tf.expand_dims(ads_cates_emb, 0) # [1, Ads, H]
    ads_cates_emb = tf.tile(ads_cates_emb, [predict_batch_size, 1, 1]) # [B, Ads, H]

    user_ads_cates_attention = attention_multi_items(ads_cates_emb, user_items_cates_emb, self.seq_len) # [B, Ads, H]
    user_ads_cates_attention = tf.layers.batch_normalization(inputs = user_ads_cates_attention, name='hist_bn', reuse=tf.AUTO_REUSE)
    # print user_sub.get_shape().as_list() 
    user_ads_cates_attention = tf.reshape(user_ads_cates_attention, [-1, hidden_units]) # [B*Ads, H]
    user_ads_cates_attention = tf.layers.dense(user_ads_cates_attention, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE) # [B*Ads, H]

    ads_cates_emb = tf.reshape(ads_cates_emb, [-1, hidden_units]) # [B*Ads, H]
    ads_din = tf.concat([user_ads_cates_attention, ads_cates_emb, 
                              user_ads_cates_attention * ads_cates_emb], axis=-1) # [B*Ads, 3H]
    ads_din = tf.layers.batch_normalization(inputs=ads_din, name='b1', reuse=True)
    ads_din_layer_1 = tf.layers.dense(ads_din, 80, activation=tf.nn.sigmoid, name='f1', reuse=True) # [B*Ads, 80]
    # ads_din_layer_1 = dice(ads_din, name='dice_1_sub')
    ads_din_layer_2 = tf.layers.dense(ads_din_layer_1, 40, activation=tf.nn.sigmoid, name='f2', reuse=True) # [B*Ads, 40]
    # ads_din_layer_2 = dice(ads_din_layer_1, name='dice_2_sub')
    ads_din_output = tf.layers.dense(ads_din_layer_2, 1, activation=None, name='f3', reuse=True) # [B*Ads, 1]
    ads_din_output = tf.reshape(ads_din_output, [-1, predict_ads_num]) # [B, Ads]

    self.logits_ads = tf.sigmoid(item_emb_b[:predict_ads_num] + ads_din_output) # [B, Ads]
    self.logits_ads = tf.reshape(self.logits_ads, [-1, predict_ads_num, 1]) # [B, Ads, 1]
    #-- fcn end -------

    self.mf_auc = tf.reduce_mean(tf.to_float(pos_neg_diff > 0))
    self.pos_score = tf.sigmoid(pos_items_emb_b + pos_din_output) # [B]
    self.neg_score = tf.sigmoid(neg_items_emb_b + neg_din_output) # [B]
    self.pos_score = tf.reshape(self.pos_score, [-1, 1]) # [B, 1]
    self.neg_score = tf.reshape(self.neg_score, [-1, 1]) # [B, 1]
    self.pos_and_neg = tf.concat([self.pos_score, self.neg_score], axis=-1) # [B, 2]

    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.label))

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                                                              self.users: uij[0],
                                                              self.pos_items: uij[1],
                                                              self.label: uij[2],
                                                              self.user_items: uij[3],
                                                              self.seq_len: uij[4],
                                                              self.lr: l,
                                                             })
    return loss


  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.pos_and_neg], feed_dict={
                                                                                self.users: uij[0],
                                                                                self.pos_items: uij[1],
                                                                                self.neg_items: uij[2],
                                                                                self.user_items: uij[3],
                                                                                self.seq_len: uij[4],
                                                                               })
    return u_auc, socre_p_and_n
  

  def test(self, sess, uij):
    return sess.run(self.logits_sub, feed_dict={
                                                self.users: uij[0],
                                                self.pos_items: uij[1],
                                                self.neg_items: uij[2],
                                                self.user_items: uij[3],
                                                self.seq_len: uij[4],
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


def attention(queries, keys, keys_length):
  '''
    queries:     [B, H]       item embeddings
    keys:        [B, T, H]    user behavior embeddings
    keys_length: [B]
  '''
  
  queries_hidden_units = queries.get_shape().as_list()[-1] # H
  queries = tf.tile(queries, [1, tf.shape(keys)[1]]) # [B, T*H]
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units]) # [B, T, H]

  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1) # [B, T, H*4]
  d_all_layer_1 = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE) # [B, T, 80]
  d_all_layer_2 = tf.layers.dense(d_all_layer_1, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE) # [B, T, 40]
  d_all_layer_output = tf.layers.dense(d_all_layer_2, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE) # [B, T, 1]
  outputs = tf.reshape(d_all_layer_output, [-1, 1, tf.shape(keys)[1]]) # [B, 1, T]

  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # [B, 1, T]
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]
  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs


def attention_multi_items(queries, keys, keys_length):
  '''
    queries:     [B, N, H] N is the number of ads
    keys:        [B, T, H] 
    keys_length: [B]
  '''

  queries_hidden_units = queries.get_shape().as_list()[-1] # H
  queries_nums = queries.get_shape().as_list()[1] # N
  queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]]) # [B, N, H*T]
  queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # [B, N, T, H]
  
  max_len = tf.shape(keys)[1] # T
  keys = tf.tile(keys, [1, queries_nums, 1]) # [B, T*N, H]
  keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # [B, N, T, H]
  
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1) # [B, N, T, 4H]
  d_all_layer_1 = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_all_layer_2 = tf.layers.dense(d_all_layer_1, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_all_layer_output = tf.layers.dense(d_all_layer_2, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE) # [B, N, T, 1]
  outputs = tf.reshape(d_all_layer_output, [-1, queries_nums, 1, max_len]) # [B, N, 1, T]

  # Mask
  key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
  key_masks = tf.tile(key_masks, [1, queries_nums]) # [B, N*T]
  key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
  outputs = tf.reshape(outputs, [-1, 1, max_len]) # [B*N, 1, T]

  keys = tf.reshape(keys, [-1, max_len, queries_hidden_units]) # [BN, T, H]
  # Weighted sum
  outputs = tf.matmul(outputs, keys) # [BN, 1, H]
  outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, H]

  return outputs