import tensorflow as tf

from Dice import dice

class Model(object):
  def __init__(self, user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num):
    self.user = tf.placeholder(tf.int32, [None,]) # [B]
    self.pos_items = tf.placeholder(tf.int32, [None,]) # [B]
    self.neg_items = tf.placeholder(tf.int32, [None,]) # [B]
    self.label = tf.placeholder(tf.float32, [None,]) # [B]
    self.user_items = tf.placeholder(tf.int32, [None, None]) # [B, T] user-item seq, T is the length of seq
    self.seq_len = tf.placeholder(tf.int32, [None,]) # [B]
    self.lr = tf.placeholder(tf.float64, []) #

    hidden_units = 128

    user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units]) # [U, H]
    item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2]) # [I, H//2]
    item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) # [I]
    cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2]) # [C, H//2]
    cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

    pos_cates = tf.gather(cate_list, self.pos_items)
    pos_item_cate_emb = tf.concat([
                                    tf.nn.embedding_lookup(item_emb_w, self.pos_items),
                                    tf.nn.embedding_lookup(cate_emb_w, pos_cates),
                                  ], axis=1)
    pos_item_b = tf.gather(item_b, self.pos_items)

    neg_cates = tf.gather(cate_list, self.neg_items)
    neg_item_cate_emb = tf.concat([
                                    tf.nn.embedding_lookup(item_emb_w, self.neg_items),
                                    tf.nn.embedding_lookup(cate_emb_w, neg_cates),
                                  ], axis=1)
    neg_item_b = tf.gather(item_b, self.neg_items)

    # get user behavior cates
    user_item_cates = tf.gather(cate_list, self.user_items)
    # get user behavior embeddings
    user_item_cate_emb = tf.concat([
                                    tf.nn.embedding_lookup(item_emb_w, self.user_items),
                                    tf.nn.embedding_lookup(cate_emb_w, user_item_cates),
                                   ], axis=2)

    pos_user_item_cates_attention = attention(pos_item_cate_emb, user_item_cate_emb, self.seq_len)
    pos_user_item_cates_attention = tf.layers.batch_normalization(inputs = pos_user_item_cates_attention)
    pos_user_item_cates_attention = tf.reshape(pos_user_item_cates_attention, [-1, hidden_units], name='hist_bn')
    pos_user_item_cates_attention = tf.layers.dense(pos_user_item_cates_attention, hidden_units, name='hist_fcn')
    pos_user = pos_user_item_cates_attention
    
    neg_user_item_cates_attention = attention(neg_item_cate_emb, user_item_cate_emb, self.seq_len)
    # neg_user_item_cates_attention = tf.layers.batch_normalization(inputs = neg_user_item_cates_attention)
    neg_user_item_cates_attention = tf.layers.batch_normalization(inputs = neg_user_item_cates_attention, reuse = True)
    neg_user_item_cates_attention = tf.reshape(neg_user_item_cates_attention, [-1, hidden_units], name='hist_bn')
    neg_user_item_cates_attention = tf.layers.dense(neg_user_item_cates_attention, hidden_units, name='hist_fcn', reuse=True)
    neg_user = neg_user_item_cates_attention

    print(pos_user.get_shape().as_list())
    print(neg_user.get_shape().as_list())
    print(pos_item_cate_emb.get_shape().as_list())
    print(neg_item_cate_emb.get_shape().as_list())

    # -- fcn begin -------
    pos_din = tf.concat([pos_user, pos_item_cate_emb, pos_user * pos_item_cate_emb], axis=-1)
    pos_din = tf.layers.batch_normalization(inputs=pos_din, name='b1')

    pos_d_layer_1 = tf.layers.dense(pos_din, 80, activation=tf.nn.sigmoid, name='f1')
    #if u want try dice change sigmoid to None and add dice layer like following two lines. You can also find model_dice.py in this folder.
    # pos_d_layer_1 = tf.layers.dense(pos_din, 80, activation=None, name='f1')
    # pos_d_layer_1 = dice(pos_d_layer_1, name='dice_1_i')
    pos_d_layer_2 = tf.layers.dense(pos_d_layer_1, 40, activation=tf.nn.sigmoid, name='f2')
    # pos_d_layer_2 = tf.layers.dense(pos_d_layer_1, 40, activation=None, name='f2')
    # pos_d_layer_2 = dice(pos_d_layer_2, name='dice_2_i')
    pos_d_output = tf.layers.dense(pos_d_layer_2, 1, activation=None, name='f3')

    neg_din = tf.concat([neg_user, neg_item_cate_emb, neg_user * neg_item_cate_emb], axis=-1)
    neg_din = tf.layers.batch_normalization(inputs=neg_din, name='b1', reuse=True)

    neg_d_layer_1 = tf.layers.dense(neg_din, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    # neg_d_layer_1 = tf.layers.dense(neg_din, 80, activation=None, name='f1', reuse=True)
    # neg_d_layer_1 = dice(neg_d_layer_1, name='dice_1_j')
    neg_d_layer_2 = tf.layers.dense(neg_d_layer_1, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    # neg_d_layer_2 = tf.layers.dense(neg_d_layer_1, 40, activation=None, name='f2', reuse=True)
    # neg_d_layer_2 = dice(neg_d_layer_2, name='dice_2_j')
    neg_d_output = tf.layers.dense(neg_d_layer_2, 1, activation=None, name='f3', reuse=True)

    pos_d_output = tf.reshape(pos_d_output, [-1])
    neg_d_output = tf.reshape(neg_d_output, [-1])

    pos_neg_diff = pos_item_b + pos_d_output - (neg_item_b + neg_d_output) # [B]
    self.logits = pos_item_b + pos_d_output

    # prediciton for selected items
    # logits for selected item:
    item_cate_emb = tf.concat([
                                item_emb_w,
                                tf.nn.embedding_lookup(cate_emb_w, cate_list)
                              ], axis=1)
    item_cate_emb_sub = item_cate_emb[:predict_ads_num,:]
    item_cate_emb_sub = tf.expand_dims(item_cate_emb_sub, 0)
    item_cate_emb_sub = tf.tile(item_cate_emb_sub, [predict_batch_size, 1, 1])

    user_sub = attention_multi_items(item_cate_emb_sub, user_item_cate_emb, self.seq_len)
    user_sub = tf.layers.batch_normalization(inputs = user_sub, name='hist_bn', reuse=tf.AUTO_REUSE)
    # print user_sub.get_shape().as_list() 
    user_sub = tf.reshape(user_sub, [-1, hidden_units])
    user_sub = tf.layers.dense(user_sub, hidden_units, name='hist_fcn', reuse=tf.AUTO_REUSE)

    item_cate_emb_sub = tf.reshape(item_cate_emb_sub, [-1, hidden_units])
    user_sub_din = tf.concat([user_sub, item_cate_emb_sub, user_sub * item_cate_emb_sub], axis=-1)
    user_sub_din = tf.layers.batch_normalization(inputs=user_sub_din, name='b1', reuse=True)
    user_sub_din_layer_1 = tf.layers.dense(user_sub_din, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
    # user_sub_din_layer_1 = dice(user_sub_din_layer_1, name='dice_1_sub')
    user_sub_din_layer_2 = tf.layers.dense(user_sub_din_layer_1, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
    # user_sub_din_layer_2 = dice(user_sub_din_layer_2, name='dice_2_sub')
    user_sub_din_output = tf.layers.dense(user_sub_din_layer_2, 1, activation=None, name='f3', reuse=True)
    user_sub_din_output = tf.reshape(user_sub_din_output, [-1, predict_ads_num])
    self.logits_sub = tf.sigmoid(item_b[:predict_ads_num] + user_sub_din_output)
    self.logits_sub = tf.reshape(self.logits_sub, [-1, predict_ads_num, 1])
    #-- fcn end -------

    
    self.mf_auc = tf.reduce_mean(tf.to_float(pos_neg_diff > 0))
    self.pos_score = tf.sigmoid(pos_item_b + pos_d_output)
    self.neg_score = tf.sigmoid(neg_item_b + neg_d_output)
    self.pos_score = tf.reshape(self.pos_score, [-1, 1])
    self.neg_score = tf.reshape(self.neg_score, [-1, 1])
    self.pos_and_neg = tf.concat([self.pos_score, self.neg_score], axis=-1)
    print(self.pos_and_neg.get_shape().as_list())


    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = \
        tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = \
        tf.assign(self.global_epoch_step, self.global_epoch_step+1)

    self.loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.label)
        )

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=self.global_step)


  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.user: uij[0],
        self.pos_items: uij[1],
        self.label: uij[2],
        self.user_items: uij[3],
        self.seq_len: uij[4],
        self.lr: l,
        })
    return loss


  def eval(self, sess, uij):
    u_auc, socre_p_and_n = sess.run([self.mf_auc, self.pos_and_neg], feed_dict={
        self.user: uij[0],
        self.pos_items: uij[1],
        self.neg_items: uij[2],
        self.user_items: uij[3],
        self.seq_len: uij[4],
        })
    return u_auc, socre_p_and_n
  

  def test(self, sess, uij):
    return sess.run(self.logits_sub, feed_dict={
        self.user: uij[0],
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
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
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
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries_nums = queries.get_shape().as_list()[1]
  queries = tf.tile(queries, [1, 1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, queries_nums, tf.shape(keys)[1], queries_hidden_units]) # shape : [B, N, T, H]
  max_len = tf.shape(keys)[1]
  keys = tf.tile(keys, [1, queries_nums, 1])
  keys = tf.reshape(keys, [-1, queries_nums, max_len, queries_hidden_units]) # shape : [B, N, T, H]
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, queries_nums, 1, max_len])
  outputs = d_layer_3_all 
  # Mask
  key_masks = tf.sequence_mask(keys_length, max_len)   # [B, T]
  key_masks = tf.tile(key_masks, [1, queries_nums])
  key_masks = tf.reshape(key_masks, [-1, queries_nums, 1, max_len]) # shape : [B, N, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, N, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, N, 1, T]
  outputs = tf.reshape(outputs, [-1, 1, max_len])
  keys = tf.reshape(keys, [-1, max_len, queries_hidden_units])
  #print outputs.get_shape().as_list()
  #print keys.get_sahpe().as_list()
  # Weighted sum
  outputs = tf.matmul(outputs, keys)
  outputs = tf.reshape(outputs, [-1, queries_nums, queries_hidden_units])  # [B, N, 1, H]
  print(outputs.get_shape().as_list())
  return outputs
