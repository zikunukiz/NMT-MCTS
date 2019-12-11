
import os

BASEPATH = '/'.join(os.getcwd().split('/')[:-1])
CODEPATH = BASEPATH + '/MultiProcess/'
MODELPATH = BASEPATH + '/savedModels/'
DATAPATH = BASEPATH + '/iwsltTokenizedData'
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
BATCHSIZE = 64 #the disjoint model is having cuda memory problems
USEBLEU1 = False
BOS_WORD_ID = 2
EOS_WORD_ID = 3
BLANK_WORD_ID = 1
