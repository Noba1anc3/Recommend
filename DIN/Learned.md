```python
with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
        df[i] = eval(line)
        i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

# eval(json_str) -> dict
# pd.DataFrame.from_dict(df, orient='index') arrange by row

pd.read_json() # read json available but changes row order

with open('../raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
# pickle.dump() save pandas DataFrame to pkl file

meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]

# DataFrame[column].unique()
# DataFrameA[column].isin(DataFrameB[column])

meta_df = meta_df.reset_index(drop=True)
# DataFrame.reset_index(drop=True) 
# if no drop, index_0 insert insert as col_0 of DataFrame
# if drops, index_0 drops

reviews_df = pd.read_pickle('../raw_data/reviews.pkl')
# pd.read_pickle(pkl_file)

reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
# DataFrame = DataFrame[[col_1, col_2, ...]]

meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
# Lambda Expression

inversed_dict = dict(zip(List, range(len(List))))
# make a inversed_dict of List {item_0: index_0, item_1: index_1}

reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
# DataFrame.sort_values([col_1, col_2])

for reviewerID, hist in reviews_df.groupby('reviewerID'):
# DataFrame.groupby(key) -> key, values(including key)

random.shuffle(train_set)
# random.shuffle(list)


```

```python
self.u = tf.placeholder(tf.int32, [None,]) # [B] user
# tf.placeholder(dtype, shape=None, name=None)

item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0)) # [I]
# tf.get_variable(name, shape, initializer)

tf.layers.batch_normalization(inputs = hist_j, reuse = True)

tf.reshape(hist_i, [-1, hidden_units], name='hist_bn')
# tf.reshape(tensor, shape, name)

tf.layers.dense(hist_i, hidden_units, name='hist_fcn')
# tf.layers.dense(inputs, units, activation, use_bias, ...)

item_cate_emb_sub = tf.expand_dims(item_cate_emb_sub, 0)
# tf.expand_dims(input, axis)

item_cate_emb_sub = tf.tile(item_cate_emb_sub, [predict_batch_size, 1, 1])
# tf.tile(input, multiples)
```

