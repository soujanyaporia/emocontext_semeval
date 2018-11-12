import argparse
import json
import pickle
import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from baseline import preprocessData, getEmbeddingMatrix


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


seed = 1234

np.random.seed(seed)
import tensorflow as tf
from tqdm import tqdm

from model import LSTM_Model

from sklearn.metrics import f1_score

tf.set_random_seed(seed)


def get_val_data():
    pass


def get_data(args):
    with open(args.config) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing test data...")
    testIndices, testTexts, u1_test, u2_test, u3_test = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(u1_train + u2_train + u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(
        u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(
        u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)
    # print(u1_trainSequences)
    print(max([len(l) for l in u1_trainSequences]))
    seq_len1 = [len(l) for l in u1_trainSequences]
    seq_len2 = [len(l) for l in u2_trainSequences]
    seq_len3 = [len(l) for l in u3_trainSequences]
    seq_len1 = np.array(seq_len1)
    seq_len2 = np.array(seq_len2)
    seq_len3 = np.array(seq_len3)
    # print(seq_len1)
    print(max(seq_len1 + seq_len2 + seq_len3))

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(trainIndices)
    l = len(trainIndices)
    sp = int(0.9 * l)
    print(sp)
    valIndices = trainIndices[sp:]
    trainIndices = trainIndices[:sp]

    train_u1_data = u1_data[trainIndices]
    train_u2_data = u2_data[trainIndices]
    train_u3_data = u3_data[trainIndices]
    val_u1_data = u1_data[valIndices]
    val_u2_data = u2_data[valIndices]
    val_u3_data = u3_data[valIndices]
    tr_sq_u1 = seq_len1[trainIndices]
    tr_sq_u2 = seq_len2[trainIndices]
    tr_sq_u3 = seq_len3[trainIndices]
    val_sq_u1 = seq_len1[valIndices]
    val_sq_u2 = seq_len2[valIndices]
    val_sq_u3 = seq_len3[valIndices]

    print(u1_data.shape)
    print(train_u1_data.shape)
    print(val_u1_data.shape)

    Y_train = labels[trainIndices]
    Y_val = labels[valIndices]
    print(Y_train.shape)

    # Perform k-fold cross validation
    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}

    print("Starting k-fold cross validation...")
    print('-' * 40)
    print("Populating embedding matrix...")
    embeddingMatrix, vocab_len, emb_size = getEmbeddingMatrix(wordIndex, EMBEDDING_DIM)
    # print('embeddingMatrix.shape', embeddingMatrix.shape)

    return train_u1_data, train_u2_data, train_u3_data, val_u1_data, val_u2_data, val_u3_data, Y_train, Y_val, \
           tr_sq_u1, tr_sq_u2, tr_sq_u3, val_sq_u1, val_sq_u2, val_sq_u3, embeddingMatrix


def main(args):
    lr = 0.001
    attn_fusion = args.attention_2
    train_u1_data, train_u2_data, train_u3_data, val_u1_data, val_u2_data, val_u3_data, Y_train, Y_val, \
    tr_sq_u1, tr_sq_u2, tr_sq_u3, val_sq_u1, val_sq_u2, val_sq_u3, embeddingMatrix = get_data(args)
    vocab_size, embedding_dim = embeddingMatrix.shape
    print('embeddingMatrix.shape', embeddingMatrix.shape)
    print('vocab_size', vocab_size)
    print('embedding_dim', embedding_dim)
    classes = Y_train.shape[-1]

    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_epoch = 0
    best_loss = 1000000.0
    best_epoch_loss = 0
    print("Building model...")
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            tf.set_random_seed(seed)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = LSTM_Model(train_u1_data.shape[1], lr, vocab_size, embedding_dim, emotions=classes,
                                   seed=seed, enable_attn_2=attn_fusion)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
                # init embeddings
                sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embeddingMatrix})

                test_feed_dict = {
                    model.input1: val_u1_data,
                    model.input2: val_u2_data,
                    model.input3: val_u3_data,
                    model.y: Y_val,
                    model.seq_len1: val_sq_u1,
                    model.seq_len2: val_sq_u2,
                    model.seq_len3: val_sq_u3,
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0

                }
                train_feed_dict = {
                    model.input1: train_u1_data,
                    model.input2: train_u2_data,
                    model.input3: train_u3_data,
                    model.y: Y_train,
                    model.seq_len1: tr_sq_u1,
                    model.seq_len2: tr_sq_u2,
                    model.seq_len3: tr_sq_u3,
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0

                }
                # print('\n\nDataset: %s' % (data))
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(train_u1_data, train_u2_data, train_u3_data, tr_sq_u1, tr_sq_u2, tr_sq_u3, Y_train)),
                        batch_size)

                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        b_train_u1_data, b_train_u2_data, b_train_u3_data, b_tr_sq_u1, b_tr_sq_u2, b_tr_sq_u3, b_Y_train = zip(
                            *batch)
                        feed_dict = {
                            model.input1: b_train_u1_data,
                            model.input2: b_train_u2_data,
                            model.input3: b_train_u3_data,
                            model.y: b_Y_train,
                            model.seq_len1: b_tr_sq_u1,
                            model.seq_len2: b_tr_sq_u2,
                            model.seq_len3: b_tr_sq_u3,
                            model.lstm_dropout: 0.5,
                            model.lstm_inp_dropout: 0.0,
                            model.dropout: 0.2,
                            model.dropout_lstm_out: 0.2

                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy, test_activations = sess.run(
                        [model.global_step, model.loss, model.accuracy, model.inter1],
                        test_feed_dict)
                    loss = loss / Y_val.shape[0]
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}".format(epoch, step, loss, accuracy))

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy

                    # if epoch == 30:
                    #     step, loss, accuracy, train_activations = sess.run(
                    #         [model.global_step, model.loss, model.accuracy, model.inter1],
                    #         train_feed_dict)

                    if loss < best_loss:
                        best_epoch_loss = epoch
                        best_loss = loss
                        # step, loss, accuracy, train_activations = sess.run(
                        # [model.global_step, model.loss, model.accuracy, model.inter1],
                        # train_feed_dict)

                print("\n\nBest epoch: {}\nBest test accuracy: {}".format(best_epoch, best_acc))
                print("\n\nBest epoch: {}\nBest test loss: {}".format(best_epoch_loss, best_loss))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--attention_2", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_raw", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--data", type=str, default='mosi')
    parser.add_argument("--classes", type=str, default='2')
    parser.add_argument('--config', help='Config to read details', required=True)

    args, _ = parser.parse_known_args(argv)

    print(args)

    batch_size = 128
    epochs = 100
    emotions = args.classes
    assert args.data in ['mosi', 'mosei', 'iemocap']

    epochs = 50
    main(args)
