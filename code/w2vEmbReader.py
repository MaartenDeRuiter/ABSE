import codecs
import logging
import numpy as np
import gensim
from sklearn.cluster import KMeans
import pickle  


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class W2VEmbReader:

    def __init__(self, args, emb_path):

        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        emb_file = codecs.open(emb_path, 'r', encoding='cp1252')#encoding='utf8')
        self.vocab_size = 0
        self.emb_dim = -1
        for line in emb_file:
            tokens = line.strip().split()
            if len(tokens) == 0:
                continue
            if self.emb_dim == -1:
                self.emb_dim = len(tokens) - 1
                assert self.emb_dim == args.emb_dim

            word = tokens[0]
            vec = tokens[1:]
            self.embeddings[word] = vec
            emb_matrix.append(vec)
            self.vocab_size += 1
        emb_file.close()

        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (len(self.embeddings), self.emb_dim))


    def get_emb_given_word(self, word):
        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_subvocab(self, subvocab):
        prior_matrix = np.empty((1, len(subvocab), 300))
        for word, index in subvocab.items():
            try:
                prior_matrix[0][index] = self.embeddings[word]
            except KeyError:
                print(word)
                pass

        #norm_prior_matrix = prior_matrix / np.linalg.norm(prior_matrix, axis=-1, keepdims=True)
        return prior_matrix

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[0][index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix
    

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float32)
    
    def get_emb_dim(self):
        return self.emb_dim
