import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs
import math
import reader as dataset
from w2vEmbReader import W2VEmbReader as EmbReader

logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)




###############################################################################################################################
## Parse arguments
#

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
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)"),
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
#parser.add_argument("-p", "--process_data", dest="process", type=str, metavar='<str>', default='False', help="Argument for preprocessing train and test data (True|False) (default=False)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain + '/' + args.rtype
U.mkdir_p(out_dir)
U.print_args(args)

batch_size = args.sen_size

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.recurrent_unit in {'lstm', 'simple'}
assert args.domain in {'res15', 'res16'}
assert args.rtype in {'sentence', 'context'}

if args.seed > 0:
    np.random.seed(args.seed)


# ###############################################################################################################################
# ## Prepare data
# #

from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

preprocess = False

train_x, train_y, train_target, test_x, test_y, test_target, \
    vocab, overal_maxlen, overal_maxlen_target, train_category, test_category \
    = dataset.prepare_data(args.domain, preprocess, args.rtype, args.fname, args.vocab_size, args.maxlen)
# Pad target sentences sequences for mini-batch processing
train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
train_target = sequence.pad_sequences(train_target, maxlen=overal_maxlen_target)
test_target = sequence.pad_sequences(test_target, maxlen=overal_maxlen_target)

# Convert y and category to categorical variables
train_y = to_categorical(train_y, 3)
train_category = to_categorical(train_category, 13)
test_y = to_categorical(test_y, 3)
test_category = to_categorical(test_category, 13)

def shuffle(array_list):
    len_ = len(array_list[0])
    for x in array_list:
        assert len(x) == len_
    p = np.random.permutation(len_)
    return [x[p] for x in array_list]

#print 'Number of training examples: ', len(train_x)
print('Length of vocab: ', len(vocab))

#def sentence_batch_generator(data, batch_size):
#    batch_count = 0
#    n_batch = len(data[0]) / batch_size
#    data = shuffle(data)
#    while True:
#        if batch_count == n_batch:
#            data = shuffle(data)
#            batch_count = 0

#        x_batch = data[0][batch_count*batch_size: (batch_count+1)*batch_size]
#        y_batch = data[1][batch_count*batch_size: (batch_count+1)*batch_size]
#        target_batch = data[2][batch_count*batch_size: (batch_count+1)*batch_size]
#        batch_count += 1


#        yield x_batch, y_batch, target_batch

def batch_generator(data, batch_size, overal_maxlen, overal_maxlentarget, neg_size = 5):
    batch_count = 0
    n_batch = int(len(data[0]) / batch_size)
    data = shuffle(data)

    while True:
        if batch_count == n_batch:
            data = shuffle(data)
            batch_count = 0


        x_batch = np.empty((0, overal_maxlen), int)
        y_batch = np.empty((0, overal_maxlen), int)
        target_batch = np.empty((0, overal_maxlentarget), int)
        x_nsamples = np.empty((0, neg_size, overal_maxlen), int)
        y_nsamples = np.empty((0, neg_size, overal_maxlen), int)
        target_nsamples = np.empty((0, neg_size, overal_maxlentarget), int)
        for b in range(batch_size):
            x_batch = np.append(x_batch, data[0][b + (batch_size * batch_count)].reshape(1, data[0].shape[1]), axis = 0) #: b + (batch_size * batch_count) + 1]
            # y_batch = np.append(y_batch, data[1][b + (batch_size * batch_count)].reshape(1, data[1].shape[1]), axis = 0) #: b + (batch_size * batch_count) + 1]
            target_batch = np.append(target_batch, data[2][b + (batch_size * batch_count)].reshape(1, data[2].shape[1]), axis = 0) #: b + (batch_size * batch_count) + 1]
            tmp_indices = []
            #for i in range(len(data[2])):
            #   if np.all(data[2][i] == data[2][b + (batch_size * batch_count)]):
            #        tmp_indices.append(i)

            #if len(tmp_indices) < 10:
            #    tmp_indices = []
            for c in range(len(data[3])):
                if np.all(data[3][c] == data[3][b + (batch_size * batch_count)]):
                    tmp_indices.append(c)

            neg_indices = np.random.choice(tmp_indices, neg_size)
            x_nsamples = np.append(x_nsamples, data[0][neg_indices].reshape(1, neg_size, data[0].shape[1]), axis=0) #.reshape(1, neg_size, data[0].shape[1]))
            #y_nsamples = np.append(y_nsamples, data[1][neg_indices].reshape(1, neg_size, data[1].shape[1]), axis = 0) #.reshape(1, neg_size, data[1].shape[1]))
            target_nsamples = np.append(target_nsamples, data[2][neg_indices].reshape(1, neg_size, data[2].shape[1]), axis=0) #.reshape(1, neg_size, data[2].shape[1]))

        batch_count += 1
        yield x_batch, y_batch, target_batch, x_nsamples, y_nsamples, target_nsamples

#def negative_batch_generator(data, batch_size, neg_size = 5):

#    # Data length 0, 1 and 2 are the same, only the dimensions differs
#    data_len = data[0].shape[0]

#    while True:
#        indices = np.random.choice(data_len, batch_size * neg_size)
#        x_samples = data[0][indices].reshape(batch_size, neg_size, data[0].shape[1])
#        print np.shape(x_samples)
#        y_samples = data[1][indices].reshape(batch_size, neg_size, data[1].shape[1])
#        target_samples = data[2][indices].reshape(batch_size, neg_size, data[2].shape[1])
#        yield x_samples, y_samples, target_samples


###############################################################################################################################
## Optimizer algorithm
#

from optimizers import get_optimizer

optimizer = get_optimizer(args)



###############################################################################################################################
## Building model

from model import create_model
import keras.backend as K
from sklearn.metrics import precision_recall_fscore_support

logger.info('  Building model')


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_model(args, vocab, overal_maxlen, overal_maxlen_target)
# freeze the word embedding layer
model.get_layer('word_emb').trainable=False
#model.get_layer('target_embs').trainable=False
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])


###############################################################################################################################
## Training
#
from keras.models import load_model
from tqdm import tqdm

logger.info('--------------------------------------------------------------------------------------------------------------------------')

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

# Generating a sample of 1 sentence as input for the model
#sen_gen = sentence_batch_generator([train_x, train_y, train_target], args.batch_size)

# Generating a sample of neg_sentences as input for the model
#neg_gen = negative_batch_generator([train_x, train_y, train_target], args.batch_size, args.neg_size)  # args.batch_size, args.neg_size)
# Batch generation of negative samples and single sentence together
batch_gen = batch_generator([train_x, train_y, train_target, train_category], args.batch_size, overal_maxlen,
                            overal_maxlen_target, args.neg_size)
batches_per_epoch = len(train_x) / args.batch_size
#batches_per_epoch = 1000

min_loss = float('inf')
for ii in range(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    for b in tqdm(range(int(batches_per_epoch))):
        #x_batch, y_batch, target_batch = sen_gen.next()
        #x_negbatch, y_negbatch, target_negbatch = neg_gen.next()
        x_batch, y_batch, target_batch, x_negbatch, y_negbatch, target_negbatch = next(batch_gen)
        x_nbatch1, y_nbatch1, target_nbatch1 = x_negbatch[:, 0, :], y_negbatch[:, 0, :], target_negbatch[:, 0, :]
        x_nbatch2, y_nbatch2, target_nbatch2 = x_negbatch[:, 1, :], y_negbatch[:, 1, :], target_negbatch[:, 1, :]
        x_nbatch3, y_nbatch3, target_nbatch3 = x_negbatch[:, 2, :], y_negbatch[:, 2, :], target_negbatch[:, 2, :]
        x_nbatch4, y_nbatch4, target_nbatch4 = x_negbatch[:, 3, :], y_negbatch[:, 3, :], target_negbatch[:, 3, :]
        x_nbatch5, y_nbatch5, target_nbatch5 = x_negbatch[:, 4, :], y_negbatch[:, 4, :], target_negbatch[:, 4, :]
        batch_loss, batch_max_margin_loss = \
            model.train_on_batch([x_batch, target_batch, x_nbatch1, target_nbatch1, x_nbatch2, target_nbatch2,
                                  x_nbatch3, target_nbatch3, x_nbatch4, target_nbatch4, x_nbatch5, target_nbatch5],
                                 np.ones((args.batch_size, 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    tr_time = time() - t0

    if loss < min_loss:
        min_loss = loss
        word_emb = model.get_layer('word_emb').get_weights()
        target_emb = model.get_layer('target_embs').get_weights()
        word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
        target_emb = target_emb / np.linalg.norm(target_emb, axis=-1, keepdims=True)
        aspect_file = codecs.open(out_dir+'/aspect.log', 'w', 'utf-8')
        model.save_weights(out_dir+'/model_param')

        for ind in range(len(target_emb[0])):
            desc = target_emb[0][ind]
            sims = word_emb.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            desc_list = [vocab_inv[w] for w in ordered_words[0][:100]]
            print('Aspect %d:' % ind)
            print(desc_list)
            aspect_file.write('Aspect %d:\n' % ind)
            aspect_file.write(' '.join(desc_list) + '\n\n')

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (loss, max_margin_loss, loss-max_margin_loss))
            
    







