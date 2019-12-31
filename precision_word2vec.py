

def precision(retrieved, relevant):
    return len(set(retrieved).intersection(relevant)) / len(retrieved)


def avarage_precision(retrieved, relevant):
    return sum(map(lambda pair: precision(retrieved[:pair[0] + 1], relevant), filter(lambda pair: pair[1] in relevant, enumerate(retrieved)))) / len(relevant)


def mean_avarage_precision(retrieveds, relevants):
    return sum(avarage_precision(retrieved, set(relevant)) for retrieved, relevant in zip(retrieveds, relevants)) / len(retrieveds)

def load(file):
    return [[tokens.split() for tokens in line.split('|')] for line in open(file)]

if __name__ == '__main__':
    data = load('pdata_word2vec.txt')
    topical, semantic, word2vec_dep, word2vec_bow = data
    print('Word2VecDEP-MAP: topical:{}, semantic:{}'.format(mean_avarage_precision(word2vec_dep, topical),
                                                    mean_avarage_precision(word2vec_dep, semantic)))
    print('Word2VecBOW-MAP: topical:{}, semantic:{}'.format(mean_avarage_precision(word2vec_bow, topical),
                                                    mean_avarage_precision(word2vec_bow, semantic)))