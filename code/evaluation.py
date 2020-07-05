import argparse
import logging
import numpy as np
from time import time
import utils as U
from sklearn.metrics import classification_report, confusion_matrix
import codecs

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='simple', help="Recurrent unit type (lstm|simple) (default=simple)")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=10000, help="Vocab size. '0' means no limit (default=10000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=3, help="The number of aspects specified by users (default=3)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("--s", dest="sen_size", type=int, metavar='<int>', default=1, help="Number of sentence instances (default=1)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=5, help="Number of negative instances (default=5)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='res16', help="domain of the corpus {res15, res16}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularization (default=0.1)")
parser.add_argument("--seed-reg", dest="seed_reg", type=float, metavar='<float>', default=10, help="The weight of seed regularization (default=10)")
parser.add_argument("--dropout-W", dest="dropout_W", type=float, metavar='<float>', default=0.5, help="The dropout of input to RNN")
parser.add_argument("--dropout-U", dest="dropout_U", type=float, metavar='<float>', default=0.1, help="The dropout of recurrent of RNN")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("-rtype", "--recon_type", dest="rtype", type=str, metavar='<str>', default='context', help="Type of reconstruction (sentence|context) (default=context)")
parser.add_argument("-fname", "--fname", dest="fname", type=str, metavar='<str>', default='../data_aspect/externalData', help="Path for data preprocessing (default=../data_aspect/externalData/domain_)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability of output layer. (default=0.5)")


args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain + '/' + args.rtype
# out_dir = '../pre_trained_model/' + args.domain
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.recurrent_unit in {'lstm', 'simple'}
assert args.domain in {'res15', 'res16'}

from keras.preprocessing import sequence
import reader as dataset

preprocess = False

###### Get test data #############
train_x, train_y, train_target, test_x, test_y, test_target, \
    vocab, overal_maxlen, overal_maxlen_target, train_category, test_category \
    = dataset.prepare_data(args.domain, preprocess, args.rtype, args.fname, args.vocab_size, args.maxlen)
train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
train_target = sequence.pad_sequences(train_target, maxlen=overal_maxlen_target)
test_target = sequence.pad_sequences(test_target, maxlen=overal_maxlen_target)

############# Build model architecture, same as the model used for training #########
from model import create_model
import keras.backend as K
from optimizers import get_optimizer

optimizer = get_optimizer(args)

def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_model(args, vocab, overal_maxlen, overal_maxlen_target)

## Load the save model parameters
model.load_weights(out_dir+'/model_param')
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])



################ Evaluation ####################################

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []


    for line in predict:
        predict_label.append(line.strip())


    for line in true:
        true_label.append(line.strip())

    cm = confusion_matrix(true_label, predict_label)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(classification_report(true_label, predict_label, ['positive', 'negative', 'neutral'], digits=3))

    #else:
    #    for line in predict:
    #        label = line.strip()
    #        if label == 'smell' or label == 'taste':
    #        label = 'taste+smell'
    #        predict_label.append(label)

    #    for line in true:
    #        label = line.strip()
    #        if label == 'smell' or label == 'taste':
    #          label = 'taste+smell'
    #        true_label.append(label)

    #    print(classification_report(true_label, predict_label,
    #        ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:,-1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)


## Create a dictionary that map word index to word
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w
test_fn = K.function([model.get_layer('sentence_input').input, model.get_layer('target_input').input, K.learning_phase()], #, K.learning_phase()
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = test_fn([test_x, test_target, 0]) #test_x, #, model.get_layer('neg_input1').input,model.get_layer('neg_input2').input, model.get_layer('neg_input3').input,model.get_layer('neg_input4').input, model.get_layer('neg_input5').input,


## Save attention weights on test sentences into a file
att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i!=0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overal_maxlen-line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' '+str(round(weights[j], 3)) + '\n')



######################################################
# Uncomment the below part for F scores
######################################################

## cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)
ind0, ind1, ind2 = 0, 0, 0
for row in aspect_probs:
    current = np.argmax(row)
    print(current)
    if current == 0:
        ind0 += 1
    if current == 1:
        ind1 += 1
    if current == 2:
        ind2 += 1

pos_ind = np.argmax([ind0,ind1,ind2])
neu_ind = np.argmin([ind0,ind1,ind2])
neg_ind = 3 - pos_ind - neu_ind
cluster_map = {}
cluster_map[pos_ind] = 'positive'
cluster_map[neg_ind] = 'negative'
cluster_map[neu_ind] = 'neutral'
print(cluster_map)


#cluster_map = {0: 'neutral', 1: 'negative', 2: 'positive'}

print('--- Results on %s domain with %s reconstruction ---' % (args.domain, args.rtype))
test_labels = '../data_aspect/%s/%s/test/polarity.txt' % (args.rtype, args.domain)
prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)


