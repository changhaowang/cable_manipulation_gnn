import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

sample_1 = [[1, 2, [3, 4, 5]],
            [3, 4, [3, 2]],
            [1, 2, [4, 5, 6, 7]]
            ]
sample_2 = [[1, 0, [3, 4]],
            [2, 0, [3, 2, 6]],
            [0, 2, [4, 7]]
            ]  
sample_3 = [[0, 2, [3, 4, 9, 0]],
            [2, 3, [3, 2, 9, 1, 0]],
            [1, 2, [4]]
            ]  
sample_data = [sample_1, sample_2, sample_3]


tfrecords_file = 'test.tfrecords'

writer = tf.io.TFRecordWriter(tfrecords_file)

for i, sample in enumerate(sample_data):

    sample_column_data = []
    # only contains 3 columns
    for _c in range(len(sample[0])):
        _col_data = []
        for _r in range(len(sample)):
            _col_data.append(sample[_r][_c])
        sample_column_data.append(_col_data)

    feature_1 = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in sample_column_data[0]
    ]

    feature_2 = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in sample_column_data[1]
    ]

    feature_3 = []  # store each list in each row
    feature_3_len = []   # store the coressponding dimension
    for row in sample_column_data[2]:
        feature_3_len.append(len(row))
        feature_3.append([tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in row])

    feature_list = {
        'dim_0': tf.train.FeatureList(feature=feature_1),
        'dim_1': tf.train.FeatureList(feature=feature_2)
    }

    context_feature = {
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
    }

    for i, (_feature, _len) in enumerate(zip(feature_3, feature_3_len)):
        feature_list['dim_2_{}'.format(str(i))] = tf.train.FeatureList(feature=_feature)
        context_feature['context_dim_2_{}'.format(str(i))] = tf.train.Feature(int64_list=tf.train.Int64List(value=[_len]))

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_feature),
        feature_lists=tf.train.FeatureLists(
            feature_list=feature_list)
    )
    serialied = example.SerializeToString()
    writer.write(serialied)

writer.close()