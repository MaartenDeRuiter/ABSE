import logging
import keras.backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
import tensorflow as tf
import numpy as np


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, vocab, overal_maxlen, overal_maxlen_target):
    from w2vEmbReader import W2VEmbReader as EmbReader
    logger.info('Initializing lookup table')
    emb_path = '../glove/%s_new.txt' % (args.domain)
    emb_reader = EmbReader(args, emb_path)

    dropout = args.dropout_W
    recurrent_dropout = args.dropout_U
    vocab_size = len(vocab)

    def ortho_reg(weight_matrix):

        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        oreg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(tf.shape(w_n)[0])))

        return args.ortho_reg*oreg #+ #seed_reg(weight_matrix)

    ##### Inputs #####
    sentence_input = Input(shape=(overal_maxlen,), dtype='int32', name='sentence_input')
    target_input = Input(shape=(overal_maxlen_target,), dtype='int32', name='target_input')
    neg_input1 = Input(shape=(overal_maxlen,), dtype='int32', name='neg_input1')
    negtarget_input1 = Input(shape=(overal_maxlen_target,), dtype='int32', name='negtarget_input1')
    neg_input2 = Input(shape=(overal_maxlen,), dtype='int32', name='neg_input2')
    negtarget_input2 = Input(shape=(overal_maxlen_target,), dtype='int32', name='negtarget_input2')
    neg_input3 = Input(shape=(overal_maxlen,), dtype='int32', name='neg_input3')
    negtarget_input3 = Input(shape=(overal_maxlen_target,), dtype='int32', name='negtarget_input3')
    neg_input4 = Input(shape=(overal_maxlen,), dtype='int32', name='neg_input4')
    negtarget_input4 = Input(shape=(overal_maxlen_target,), dtype='int32', name='negtarget_input4')
    neg_input5 = Input(shape=(overal_maxlen,), dtype='int32', name='neg_input5')
    negtarget_input5 = Input(shape=(overal_maxlen_target,), dtype='int32', name='negtarget_input5')


    ##### Construct word embedding layer #####
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')

    ##### Experiment with seed_regularization #####
    def seed_reg(weight_matrix):
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=0, keepdims=True)),
                                     K.floatx())
        vocab_pos = {'good': 0, 'great': 1, 'nice': 2, 'impecable': 3, 'excellent': 4}
        vocab_neg = {'gross': 0, 'bad': 1, 'terrible': 2, 'aweful': 3, 'horrible': 4}
        vocab_neu = {'mediocre': 0, 'reasonable': 1, 'decent': 2, 'average': 3, 'ok': 4}
        prior_matrix = np.empty((0, len(vocab_pos), args.emb_dim))
        sub_vocab = [vocab_pos, vocab_neg, vocab_neu]
        for sub in sub_vocab:
            prior_matrix = np.append(prior_matrix, emb_reader.get_emb_matrix_subvocab(sub), axis=0)

        r_pos = np.sum(prior_matrix[0], axis=0) / np.linalg.norm(np.sum(prior_matrix[0], axis=0), axis=-1,
                                                                 keepdims=True)
        r_neg = np.sum(prior_matrix[1], axis=0) / np.linalg.norm(np.sum(prior_matrix[1], axis=0), axis=-1,
                                                                 keepdims=True)
        r_neu = np.sum(prior_matrix[2], axis=0) / np.linalg.norm(np.sum(prior_matrix[2], axis=0), axis=-1,
                                                                 keepdims=True)

        sreg = K.sum(1. - r_pos * w_n[0] - r_neg * w_n[1] - r_neu * w_n[2], axis=-1)
        return args.seed_reg*sreg

    ##### Represent target as averaged word embedding #####
    print('Use average term embeddings as target embedding')
    target_term_embs = word_emb(target_input)
    negtarget_term_embs1 = word_emb(negtarget_input1)
    negtarget_term_embs2 = word_emb(negtarget_input2)
    negtarget_term_embs3 = word_emb(negtarget_input3)
    negtarget_term_embs4 = word_emb(negtarget_input4)
    negtarget_term_embs5 = word_emb(negtarget_input5)
    target_embs = Average(mask_zero=True, name='target_emb')(target_term_embs)
    negtarget_embs1 = Average(mask_zero=True, name='target_embs_n1')(negtarget_term_embs1)
    negtarget_embs2 = Average(mask_zero=True, name='target_embs_n2')(negtarget_term_embs2)
    negtarget_embs3 = Average(mask_zero=True, name='target_embs_n3')(negtarget_term_embs3)
    negtarget_embs4 = Average(mask_zero=True, name='target_embs_n4')(negtarget_term_embs4)
    negtarget_embs5 = Average(mask_zero=True, name='target_embs_n5')(negtarget_term_embs5)


    ##### Obtain word embeddings #####
    sentence_output = word_emb(sentence_input)
    neg_output1 = word_emb(neg_input1)
    neg_output2 = word_emb(neg_input2)
    neg_output3 = word_emb(neg_input3)
    neg_output4 = word_emb(neg_input4)
    neg_output5 = word_emb(neg_input5)


    ##### LSTM layer #####
    print('Use LSTM layer')
    lstm = LSTM(args.rnn_dim, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout, name='lstm')
    sentence_output = lstm(sentence_output)
    neg_output1 = lstm(neg_output1)
    neg_output2 = lstm(neg_output2)
    neg_output3 = lstm(neg_output3)
    neg_output4 = lstm(neg_output4)
    neg_output5 = lstm(neg_output5)

    ##### Compute sentence representation #####
    print('Attention layer target-based sentence representation')
    att_weights = Attention(name='att_weights')([sentence_output, target_embs])
    z_s = WeightedSum()([sentence_output, att_weights])

    ##### Compute representations of negative instances #####
    print('Attention layer negative target-based sentence representation samples')
    negatt_weights1 = Attention(name='att_weights_n1')([neg_output1, negtarget_embs1])
    negatt_weights2 = Attention(name='att_weights_n2')([neg_output2, negtarget_embs2])
    negatt_weights3 = Attention(name='att_weights_n3')([neg_output3, negtarget_embs3])
    negatt_weights4 = Attention(name='att_weights_n4')([neg_output4, negtarget_embs4])
    negatt_weights5 = Attention(name='att_weights_n5')([neg_output5, negtarget_embs5])
    z_n1 = WeightedSum()([neg_output1, negatt_weights1])
    z_n2 = WeightedSum()([neg_output2, negatt_weights2])
    z_n3 = WeightedSum()([neg_output3, negatt_weights3])
    z_n4 = WeightedSum()([neg_output4, negatt_weights4])
    z_n5 = WeightedSum()([neg_output5, negatt_weights5])

    if args.dropout_prob > 0:
        print('use dropout layer')
        z_s = Dropout(args.dropout_prob)(z_s)
        z_n1 = Dropout(args.dropout_prob)(z_n1)
        z_n2 = Dropout(args.dropout_prob)(z_n2)
        z_n3 = Dropout(args.dropout_prob)(z_n3)
        z_n4 = Dropout(args.dropout_prob)(z_n4)
        z_n5 = Dropout(args.dropout_prob)(z_n5)

    ##### Reconstruction #####
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='target_embs', W_regularizer=ortho_reg)(p_t)

    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, z_n1, z_n2, z_n3, z_n4, z_n5, r_s])
    model = Model(inputs=[sentence_input, target_input, neg_input1, negtarget_input1, neg_input2, negtarget_input2,
                         neg_input3, negtarget_input3, neg_input4, negtarget_input4, neg_input5, negtarget_input5],
                  outputs=loss)

    logger.info('  Done')

    logger.info('Initializing word embedding matrix')
    model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
    model.get_layer('target_embs').set_weights([emb_reader.get_aspect_matrix(args.aspect_size)])

    return model