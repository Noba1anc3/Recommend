import numpy as np

class DataInput:
  def __init__(self, data, batch_size):
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def next(self):
    if self.i == self.epoch_size:
      raise StopIteration

    batch_data = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    user, pos_item, label, seq_length = [], [], [], []
    for data in batch_data:
      user.append(data[0])
      pos_item.append(data[1])
      label.append(data[4])
      seq_length.append(data[3])

    max_seq_len = max(seq_length)
    user_items = np.zeros([len(batch_data), max_seq_len], np.int64)

    k = 0
    for data in batch_data:
      for item_idx in range(data[3]):
        user_items[k][item_idx] = data[2][item_idx]
      k += 1

    return self.i, (user, pos_item, label, user_items, seq_length)


class DataInputTest:
  def __init__(self, data, batch_size):
    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def next(self):
    if self.i == self.epoch_size:
      raise StopIteration

    batch_data = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
    self.i += 1

    user, pos_item, neg_item, seq_len = [], [], [], []
    for data in batch_data:
      user.append(data[0])
      pos_item.append(data[1][0])
      neg_item.append(data[1][1])
      seq_len.append(data[3])

    max_sl = max(seq_len)
    user_items = np.zeros([len(batch_data), max_sl], np.int64)

    k = 0
    for data in batch_data:
      for item_idx in range(data[3]):
        user_items[k][item_idx] = data[2][item_idx]
      k += 1

    return self.i, (user, pos_item, neg_item, user_items, seq_len)
