import tensorflow as tf 
import tensorflow.keras as tf 


def get_training_data(dataset): 
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32')
    X_train = X_train/255.

    X_test = X_test.astype('float32')
    X_test = X_test/255.

    y_train = tf.reshape(tf.one_hot(y_train, 10), shape=(-1, 10))
    y_test = tf.reshape(tf.one_hot(y_test, 10), shape=(-1, 10))

    BATCH_SIZE = 256
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1024).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, test_ds