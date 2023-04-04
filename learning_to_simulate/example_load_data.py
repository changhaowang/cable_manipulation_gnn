import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

tfrecords_file = 'test.tfrecords'
context_dict = {'index': tf.FixedLenFeature([], dtype=tf.int64)}
features_dict = {'dim_0': tf.VarLenFeature(dtype=tf.int64),
                 'dim_1': tf.VarLenFeature(dtype=tf.int64)
                 }

_window_size = 3
for i in range(_window_size):
    features_dict['dim_2_{}'.format(str(i))] = tf.VarLenFeature(dtype=tf.int64)
    context_dict['context_dim_2_{}'.format(str(i))] = tf.FixedLenFeature([], dtype=tf.int64)

def parse_tfrecord(example):
    context, features = tf.parse_single_sequence_example(
        example, sequence_features=features_dict, context_features=context_dict)

    index = context['index']
    context_dim_2 = [context['context_dim_2_{}'.format(str(i))] for i in range(_window_size)]
    # import ipdb; ipdb.set_trace(context=20)

    dim_0 = tf.sparse_tensor_to_dense(features['dim_0'])
    dim_1 = tf.sparse_tensor_to_dense(features['dim_1'])
    dim_2 = [
        tf.sparse_tensor_to_dense(features['dim_2_{}'.format(str(i))]) for i in range(_window_size)
    ]
    return (index, *context_dim_2, dim_0, dim_1,  *dim_2)

Dataset = tf.data.TFRecordDataset(tfrecords_file)
Dataset = Dataset.map(parse_tfrecord)
iterator = Dataset.make_one_shot_iterator()
# with tf.Session() as sess:
tf_data = []
for _i in range(_window_size):
    tf_data.append(iterator.get_next())