import itertools, os, json
import struct, copy, random
import numpy as np

with open('config.json') as config_file:
    config = json.load(config_file)

DATA_FD = config["data_folder"]
nLogKeys = 50

class Data:

    def __init__(self, bTrainAdv=0, dataType='fake', bTrain = True): # if bTrain==True, do not load test data, to save memory
        self.dataType = dataType

        self.histLen = config['hist_len']
        self.data_split = {'train':0.95, 'valid':0.05}

        if self.dataType == 'hdfs':
            self.TRAIN_FILE = '../datasets/hdfs/normalBlk_allREs_10w.logSeqs'  #'normalBlk_allREs_10w.logSeqs'
            self.TEST_FILE = ''
        dataX, dataY = self.getLogData(self.TRAIN_FILE)
        num_train = int(len(dataX) * self.data_split['train'])

        self.x_train, self.y_train = dataX[0:num_train], dataY[0:num_train]
        self.x_valid, self.y_valid = dataX[num_train:], dataY[num_train:]


    def seq2xy(self, inSeq):
        dataX, dataY = [], []
        iLen = len(inSeq)
        print('self.key_to_oneHot', self.key_to_oneHot)
        for i in range(0, iLen - self.histLen, 1):
            seq_in = inSeq[i : i+self.histLen]
            seq_out = inSeq[i+self.histLen]
            dataX.append([self.key_to_oneHot[k] for k in seq_in])
            dataY.append(self.key_to_int[seq_out])

        return dataX, dataY


    def getLogData(self, logFile):
        text_seqs = []
        if self.dataType == 'bgl':
            with open(self.TRAIN_FILE) as fp:
                oneSeq, bNormal = [], 1
                for ln in fp.readlines():
                    tmp = ln.strip().split()
                    if tmp[2][-3:] == '100':
                        if bNormal: text_seqs.append(oneSeq)
                        oneSeq = []
                    else:
                        key = int(tmp[0])
                        bNormal *= int(tmp[1])
                        oneSeq.append(key)
                if bNormal: text_seqs.append(oneSeq) # the last one
            print('# of seqs/clients', len(text_seqs))
        else: # for hdfs and openstack
            print('current dir: ', os.getcwd())
            for ll in open(self.TRAIN_FILE).readlines(): text_seqs.append([ii for ii in ll.split()])  # int for sorting

        text_seqs_all = list(itertools.chain.from_iterable(text_seqs))
        trainLogKeys = sorted(list(set(text_seqs_all)))
        # keyLen = len(logKeys)
        allLogKeys = [str(i) for i in range(nLogKeys)]
        key_to_int = dict((str(i), i) for i in range(nLogKeys)) #dict((c, i) for i, c in enumerate(logKeys))
        key_to_oneHot = {}
        self.NUM_CLASSES = len(allLogKeys)+1
        print("NUM_CLASSES", self.NUM_CLASSES)
        for i, k in enumerate(allLogKeys):
            tmp = [0 for _ in range(self.NUM_CLASSES)]
            tmp[i] = 1
            key_to_oneHot[k] = tmp

        self.trainLogKeys, self.allLogKeys, self.key_to_int, self.key_to_oneHot = trainLogKeys, allLogKeys, key_to_int, key_to_oneHot

        dataX, dataY = [], []
        for iSeq in text_seqs:
            iLen = len(iSeq)
            for i in range(0, iLen - self.histLen, 1):
                seq_in = iSeq[i : i+self.histLen]
                seq_out = iSeq[i+self.histLen]
                dataX.append([key_to_oneHot[k] for k in seq_in])
                dataY.append(key_to_int[seq_out])

        # random shuffle the dataset (in order to randomly select valid dataset)
        idxs = range(len(dataX))
        random.shuffle(idxs) # return None
        dataX = np.array(dataX)[idxs]
        dataY = np.array(dataY)[idxs]

        dataX, dataY = list(dataX), list(dataY)
        return dataX, dataY
