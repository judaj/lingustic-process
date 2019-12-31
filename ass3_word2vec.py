import sys
import numpy as np
import pandas as pd

"""
Yehuda Gihasi, 305420671
Elron Bandel, 308130038
"""


def print_top20_to_file(word2VecDep, word2VecBow5):
    outputFile_top20 = open('top20_Word2Vec.txt', 'w', encoding='utf-8')
    for word in word2VecDep.wordsToCheckSimilarity:
        outputFile_top20.write(word + "\n")
        for x in range(len(word2VecDep.similarWords[word])):
            string = word2VecDep.similarWords[word][x] + " " + word2VecBow5.similarWords[word][x] + "\n"
            outputFile_top20.write(string)
        outputFile_top20.write("*********\n")

    outputFile_top20.close()


def print_top10contexts_to_file(word2VecDep, word2VecBow5):
    outputFile_top10contexts = open('top10contexts_Word2Vec.txt', 'w', encoding='utf-8')
    for word in word2VecDep.wordsToCheckSimilarity:
        outputFile_top10contexts.write(word + "\n")
        wordVecDep = word2VecDep.word2Vector[word]
        wordVecBow5 = word2VecBow5.word2Vector[word]

        similar_words_dep = [word[0] for word in word2VecDep.calculate_similar_vector_contex(wordVecDep)[:11]]
        similar_words_bow5 = [word[0] for word in word2VecDep.calculate_similar_vector_contex(wordVecBow5)[:11]]

        for x in range(len(similar_words_dep)):
            string = similar_words_dep[x] + " " + similar_words_bow5[x] + "\n"
            outputFile_top10contexts.write(string)
        outputFile_top10contexts.write("*********\n")

    outputFile_top10contexts.close()


class Word2vec:
    def __init__(self):
        self.word2Vector = {}
        self.similarWords = {}
        self.vectors = []
        self.words = []
        self.wordsToCheckSimilarity = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']


    def convert_word_2_vec(self):
        word2vec = {self.words[x][0]: x for x in range(len(self.words))}

        for word in self.wordsToCheckSimilarity:
            self.word2Vector[word] = self.vectors[word2vec[word]]


    def search_similar_words(self):
        for word in self.wordsToCheckSimilarity:
            similarWords = [word[0] for word in self.calculate_similar_vector_contex(self.word2Vector[word])]
            self.similarWords[word] = similarWords[1:21] # index 0 its the word himself


    def calculate_similar_vector_contex(self, wordVec):
        dotProduct = self.vectors.dot(wordVec)
        similarTop10 = dotProduct.argsort()[-1:10:-1]
        return self.words[similarTop10]


    def load_words_and_matrix(self, fName):
        self.words = pd.read_csv(fName, header=None, delimiter=' ', dtype=str, usecols=[0]).values

        with open(fName, 'r') as f:
            numberColums = len(f.readline().split())
        numberRows = sum(1 for _ in open(fName, encoding='utf-8'))
        numberColums -= 1
        self.vectors = []
        self.vectors = np.empty([numberRows, numberColums], dtype=np.float32)
        i = 0
        with open(fName, 'r', encoding='utf-8') as f:
            for line in f:
                self.vectors[i] = line.rstrip().split()[1:]
                i += 1


if __name__ == '__main__':
    wordFNameDep, contextsFNameDep = sys.argv[1], sys.argv[2]
    wordFNameBow5, contextsFNameBow5 = sys.argv[3], sys.argv[4]
    word2VecDep = Word2vec()
    word2VecDep.load_words_and_matrix(wordFNameDep)
    word2VecDep.convert_word_2_vec()
    word2VecDep.search_similar_words()
    word2VecDep.load_words_and_matrix(contextsFNameDep)


    word2VecBow5 = Word2vec()
    word2VecBow5.load_words_and_matrix(wordFNameBow5)
    word2VecBow5.convert_word_2_vec()
    word2VecBow5.search_similar_words()
    word2VecBow5.load_words_and_matrix(contextsFNameBow5)


    print_top20_to_file(word2VecDep, word2VecBow5)
    print_top10contexts_to_file(word2VecDep, word2VecBow5)
