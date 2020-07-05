import codecs
import re
import operator
#from itertools import izip
#import zip
import numpy as np
from dataReader import read_data_xml

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def is_number(token):
    return bool(num_regex.match(token))

def create_vocab(domain, rtype, maxlen=0, vocab_size=0):
    assert domain in {'res15', 'res16'}
    assert rtype in {'sentence', 'context'}

    #source = '../preprocessed_data/'+domain+'/train.txt'
    file_list = ['../data_aspect/%s/%s/train/sentence.txt'%(rtype, domain),
                 '../data_aspect/%s/%s/test/sentence.txt'%(rtype, domain)]

    print('Creating vocabulary...')

    total_words, unique_words = 0, 0
    word_freqs = {}

    for f in file_list:
        top = 0
        fin = codecs.open(f, 'r', 'utf-8')
        for line in fin:
            words = line.split()
            if maxlen > 0 and len(words) > maxlen:
                continue
            for w in words:
                if not is_number(w):
                    try:
                        word_freqs[w] += 1
                    except KeyError:
                        unique_words += 1
                        word_freqs[w] = 1
                    total_words += 1

    print ('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>':0, '<unk>':1, '<num>':2}
    index = len(vocab)
    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print (' keep the top %i words' % vocab_size)

    #Write (vocab, frequence) to a txt file
    print('Vocab for' + domain + rtype)
    vocab_file = codecs.open(domain+'_vocab_'+rtype, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word+'\t'+str(0)+'\n')
            continue
        vocab_file.write(word+'\t'+str(word_freqs[word])+'\n')
    vocab_file.close()

    #Write vocab to a txt file
    # vocab_file = codecs.open(domain+'_vocab', mode='w', encoding='utf8')
    # sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    # for word, index in sorted_vocab:
    #     vocab_file.write(word+'\t'+str(index)+'\n')
    # vocab_file.close()

    return vocab

def read_dataset_target(domain, phase, rtype, vocab, maxlen):
    assert domain in ['res15', 'res16']
    assert phase in ['train', 'test']
    assert rtype in ['sentence', 'context']

    print('Preparing dataset...')
    data_x, data_y, target, data_cat = [], [], [], []
    polarity_category = {'positive': 0, 'negative': 1, 'neutral': 2}
    target_category = {'AMBIENCE#GENERAL': 0, 'DRINKS#PRICES': 1, 'DRINKS#QUALITY': 2,
                'DRINKS#STYLE_OPTIONS': 3, 'FOOD#PRICES': 4, 'FOOD#QUALITY': 5,
                'FOOD#STYLE_OPTIONS': 6, 'LOCATION#GENERAL': 7, 'RESTAURANT#GENERAL': 8,
                'RESTAURANT#MISCELLANEOUS': 9, 'RESTAURANT#PRICES': 10, 'SERVICE#GENERAL': 11,
                       'FOOD#GENERAL': 12}
    file_names = ['../data_aspect/%s/%s/%s/sentence.txt' % (rtype, domain, phase),
                  '../data_aspect/%s/%s/%s/polarity.txt' % (rtype, domain, phase),
                  '../data_aspect/%s/%s/%s/term.txt' % (rtype, domain, phase),
                  '../data_aspect/%s/%s/%s/category.txt' % (rtype, domain, phase)]

    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    maxlen_target = 0

    files = [open(i, 'r') for i in file_names]
    for rows in zip(*files): #zip(*files):
        content = rows[0].strip().split()
        polarity = rows[1].strip()
        target_content = rows[2].strip().split()
        category = rows[3].split()



        if maxlen > 0 and len(words) > maxlen:
            continue

        content_indices = []
        if len(content) == 0:
            content_indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in content:
            if is_number(word):
                content_indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                content_indices.append(vocab[word])
            else:
                content_indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(content_indices)
        data_y.append(polarity_category[polarity])
        data_cat.append(target_category[str(category[0])])

        target_indices = []
        if len(target_content) == 0:
            target_indices.append(vocab['<unk>'])
            unk_hit += 1
        for word in target_content:
            if is_number(word):
                target_indices.append(vocab['<num>'])
            elif word in vocab:
                target_indices.append(vocab[word])
            else:
                target_indices.append(vocab['<unk>'])
        target.append(target_indices)

        if maxlen_x < len(content_indices):
            maxlen_x = len(content_indices)
        if maxlen_target < len(target_indices):
            maxlen_target = len(target_indices)

    print('Reader lengths', maxlen_x, maxlen_target)
    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, data_y, target, maxlen_x, maxlen_target, data_cat

def get_data_target(vocab, domain, rtype, maxlen=0):
    assert domain in ['res15', 'res16']

    train_x, train_y, train_target, train_maxlen, train_maxlen_target, train_category = read_dataset_target(domain, 'train', rtype, vocab, maxlen)
    test_x, test_y, test_target, test_maxlen, test_maxlen_target, test_category = read_dataset_target(domain, 'test', rtype, vocab, maxlen)
    overal_maxlen = max(train_maxlen, test_maxlen)
    overal_maxlen_target = max(train_maxlen_target, test_maxlen_target)

    print('-Overal_maxlen: ', overal_maxlen)
    print('-Overal_maxlen_aspect: ', overal_maxlen_target)

    return train_x, train_y, train_target, test_x, test_y, test_target, overal_maxlen, overal_maxlen_target, train_category, test_category



def prepare_data(domain, process, rtype, fname, vocab_size=0, maxlen=0):
    print('Reading data from', domain)
    print('-Creating vocab ...')

    if process == True:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}
        print('Pre-processing training data...')
        read_data_xml(fname, source_count, source_word2idx, target_count, target_phrase2idx,
                      domain, 'train', rtype)
        print('Pre-processing test data...')
        read_data_xml(fname, source_count, source_word2idx, target_count, target_phrase2idx,
                      domain, 'test', rtype)
    vocab = create_vocab(domain, rtype, maxlen, vocab_size)

    print('-Reading dataset ...')
    print('--train set and test set')
    train_x, train_y, train_aspect, test_x, test_y, test_aspect, \
        overal_maxlen, overal_maxlen_target, train_category, test_category = get_data_target(vocab, domain, rtype)

    return train_x, train_y, train_aspect, test_x, test_y, test_aspect, vocab, overal_maxlen, overal_maxlen_target, train_category, test_category
    


#if __name__ == "__main__":
#    vocab, train_x, test_x, maxlen = get_data('restaurant')
#    print len(train_x)
#    print len(test_x)
#    print maxlen

