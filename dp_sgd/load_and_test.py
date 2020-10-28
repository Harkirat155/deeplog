from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import lstm_inference_dp as lstm_inference
import math, os, json, sys, copy
from data_reader import Data
from utils import *
import datetime, argparse

np.random.seed(1)
tf.set_random_seed(123)

bUsePrevThres = False

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--sampleRate','-sr', default=0.1, type=float, help='sample percentage of full hdfs data')
parser.add_argument('--start','-st', default=0, type=int, help='start line id of the input data') # for debugging only
parser.add_argument('--end','-e', default=-1, type=int, help='end line id of the input data') # for debugging only
parser.add_argument('--learnRate','-lr', default=0.0005, type=float, help='unlearning rate')
parser.add_argument('--maxCnt','-mc', default=20, type=float, help='maximum iterations of learning for a single sample')
parser.add_argument('--thresPrev','-t_pv', default=0.9999, type=float, help='anomaly threshold for prev')
parser.add_argument('--thresPred','-t_pd', default=0.0001, type=float, help='anomaly threshold for prediction')
parser.add_argument('--model','-m', default=None, type=str, help='model to be used for detection')

args = parser.parse_args()

print('==========args.model', args.model)

with open('config.json') as config_file:
    config = json.load(config_file)
DATA_FD = config["data_folder"]
RESULT_FD = config["result_folder"]
MODEL_FD = config["model_folder"]
logType = config['data_type']

NORMAL, ABNORMAL = 1, 0
if logType=='android':
    NORMAL, ABNORMAL = 1, 0

normalSamplePerc = args.sampleRate # 0.1, 0.01
startCnt= args.start #1300 #1500
keepCnt = args.end  #2000 #-1


FPpreds = {} # only for stats


learnRate = args.learnRate
# else
DATA = Data(bTrainAdv=0, dataType=logType, bTrain = True)
train_op, eval_correct, loss, data_placeholder, label_placeholder, pred = \
        lstm_inference.lstm_model(oneHot=True, bTrain=False, num_classes=DATA.NUM_CLASSES)

filteredSeqs = [os.path.join(DATA_FD, 'hdfs/abnormalBlk_normalKey(10w)AndLen.logSeqs'), \
                    os.path.join(DATA_FD, 'hdfs/normalBlk_sample%s_10w-end.logSeqs'%(str(normalSamplePerc)))]
filteredLogsDetectRsts_seq = os.path.join(RESULT_FD, 'hdfs/mixedBlk.sampleNormal%s.logSeqs.withLabels.detectRsts'%(str(normalSamplePerc)))
mixedSeqFile = os.path.join(DATA_FD, 'hdfs/mixedBlk.sampleNormal%s.logSeqs.withLabels'%(str(normalSamplePerc)))
print('mixedSeqFile', mixedSeqFile)

postfix = '_sr'+str(args.sampleRate) + '_start'+str(args.start)+'_end'+str(args.end)+'_lr'+str(args.learnRate)

if bUsePrevThres: postfix += '_thresPrev'+str(args.thresPrev)
else:  postfix += '_thresPred'+str(args.thresPred)

filteredLogsDetectRsts_seq += postfix

rst_fd = os.path.dirname(filteredLogsDetectRsts_seq)
print('rst_fd', rst_fd)
try:
    os.makedirs(rst_fd)
except OSError:
    if not os.path.isdir(rst_fd):
        raise


def getOrderedSeqsAndLables():
    global filteredSeqs
    orderedBlks, allSeqs, allLabels = [], [], []
    print('mixedSeqFile to check', mixedSeqFile)
    if os.path.exists(mixedSeqFile):
        with open(mixedSeqFile) as fp:
            for ln in fp:
                tmp = ln.strip().split()
                allLabels.append(int(tmp[0]))
                allSeqs.append(tmp[1:])
        print('seq cnt', len(allSeqs), len(allLabels))
        #print('startCnt:keepCnt', startCnt, keepCnt, 'allSeqs[startCnt:keepCnt]', allSeqs[startCnt:keepCnt])
        return allSeqs[startCnt:keepCnt], allLabels[startCnt:keepCnt]
        #return allSeqs, allLabels

    print('============MIXED FILE DOES NOT EXIST, GENERATE NEW!!!================')

# else
    blkOrderFile = os.path.join(DATA_FD, 'hdfs/nameIndex.txt')
    with open(blkOrderFile) as fp:
        orderedBlks = [line.strip() for line in fp]
# normal blks
    normalBlk_seq = {}
    with open(filteredSeqs[1]) as fp:
        for ln in fp:
            tmp = ln.strip().split()
            normalBlk_seq[tmp[-1]] = tmp
# abnormal blks
    abnormalBlk_seq = {}
    fpSeq = open(mixedSeqFile, 'w')
    with open(filteredSeqs[0]) as fp:
        for ln in fp:
            tmp = ln.strip().split()
            abnormalBlk_seq[tmp[-1]] = tmp
    for ib, blk in enumerate(orderedBlks):
        print('ib', ib)
        if blk in normalBlk_seq.keys():
            allSeqs.append(normalBlk_seq[blk])
            allLabels.append(1)
            fpSeq.write(str(allLabels[-1])+'\t'+' '.join(allSeqs[-1])+'\n')
        elif blk in abnormalBlk_seq.keys():
            allSeqs.append(abnormalBlk_seq[blk])
            allLabels.append(0)
            fpSeq.write(str(allLabels[-1])+'\t'+' '.join(allSeqs[-1])+'\n')
    fpSeq.close()
    # write to file
    # for i, label in enumerate(allLabels):
    #     with open(mixedSeqFile, 'w') as fp:
    #         fp.write(str(label)+'\t'+' '.join(allSeqs[i])+'\n')
    print('complete ordered seqs')
    return allSeqs[:keepCnt], allLabels[:keepCnt]


def eval_all(x_test, y_test, sess):
    feed_dict = {str(data_placeholder.name): np.asarray(DATA.x_test),
                 str(label_placeholder.name): np.asarray(DATA.y_test)}
    global_loss = sess.run(loss, feed_dict=feed_dict)
    count = sess.run(eval_correct, feed_dict=feed_dict)
    accuracy = float(count) / float(len(y_test))
    print('cur model total loss %f, acc %f'%(global_loss, accuracy))

def getPreds(sess, x, y):
    prediction = sess.run(pred, feed_dict= {str(data_placeholder.name): x})
    prediction = np.squeeze(prediction)
    #print('prediction', prediction)
    #print('in getPreds, y', y)

    pos, prevProb = getIndexAndPrevProb(prediction[y], prediction) # todo
    return pos, prevProb, prediction[y]

def isDetectedNormal(prevProb, predProb, nLogSeqs, trueLabel, pos):
    print('nLogSeqs:%d, trueLabel:%s, bUsePrevThres:%s, prevProb:%s, predProb:%s, pos:%d'%(nLogSeqs, str(trueLabel), str(bUsePrevThres), str(prevProb), str(predProb), pos))
    if bUsePrevThres and prevProb >= args.thresPrev or not bUsePrevThres and predProb <= args.thresPred:
        return False
    return True

def eval(load_file):
# begin loading model
    print('evaluating', load_file)
    model_dir = os.path.dirname(load_file)
    model_file = os.path.basename(load_file)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('# layers eval_all 01:', len(tf.trainable_variables()))

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('# layers eval_all 011:', len(tf.trainable_variables()))
    saver = tf.train.import_meta_graph(load_file + '.meta')
    print('# layers eval_all 012:', len(tf.trainable_variables()))
    saver.restore(sess, load_file)
    print('loaded model...', load_file)
# end loading model

    allSeqs, allLabels = getOrderedSeqsAndLables()
    with open(filteredLogsDetectRsts_seq, 'w') as fp_seq_rsts:
        for ii in [0]:
            nNORMAL, nABNORMAL, nAnomalies = 0, 0, 0
            FP_byLog = []
            FN_byLog = []
            nLogSeqs = 0
            prevFNs, prevFPs = [], []
            for ii, seq in enumerate(allSeqs):
#                print('ii', ii)
                test_seqs = seq
                if logType=='hdfs': test_seqs = seq[:-1]
                trueLabel = allLabels[ii]
                if trueLabel==NORMAL: nNORMAL += 1
                else: nABNORMAL += 1

                nLogSeqs += 1
                minPredProb, maxPrevProb, pos_marked, x_marked, y_marked, hist_marked, pred_marked = \
                                                        1, 0, None, None, None, None, None
                bWARN = False
                toWrite_detecRsts_seq = []
                pattern = []
                for k in test_seqs[:DATA.histLen]:
                    pattern.append(k)
                    toWrite_detecRsts_seq.append("%s:f"%(k))

                for i in range(DATA.histLen, len(test_seqs)):
                    x =[]
                    bSkip = False
                    for k in pattern:
                        if k in DATA.key_to_oneHot: x.append(DATA.key_to_oneHot[k])
                        else:
                            print('XXXXXXXXXXXXXXXXXXXXXXXXXX: unknown key', k, 'normal?', trueLabel==NORMAL)
                            bSkip = True
                            break
#                    x =[DATA.key_to_oneHot[k] for k in pattern]
                    if bSkip: 
                        if trueLabel==ABNORMAL: bWARN = True
                        continue
                    x = np.expand_dims(x, axis=0)
                    #y = DATA.key_to_int[test_seqs[i]]
                    if test_seqs[i] in  DATA.key_to_int: y = DATA.key_to_int[test_seqs[i]]
                    else: 
                        print('XXXXXXXXXXXXXXXXXXXXXXXXXX: unknown key', k, 'normal?', trueLabel==NORMAL)
                        continue
    
                    pos, prevProb, predProb = getPreds(sess, x, y)

    ######                  # check prevProb and pos here. count false positives and false negatives
                    rst = ""
                    if not isDetectedNormal(prevProb, predProb, nLogSeqs, trueLabel, pos):
#                        print('-------anomaly detected, pos: %d, prevProb: %.5f, predProb: %.5f'%(pos, prevProb, predProb), 'x', pattern, 'y', test_seqs[i])
                        bWARN = True
                        if trueLabel == NORMAL:
                            FP_byLog.append(nLogSeqs)
                        rst="a_"
                    else: rst = "n_"
                    rst+= "pos%02d_prevProb%.5f"%(pos, prevProb)
                    toWrite_detecRsts_seq.append("%s:%s"%(test_seqs[i], rst))
    ######
                    if bUsePrevThres and prevProb > maxPrevProb or not bUsePrevThres and predProb < minPredProb: #not isDetectedNormal(prevProb, predProb):
                        minPredProb, maxPrevProb, pos_marked, x_marked, y_marked, hist_marked, pred_marked = \
                                predProb, prevProb, pos, copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(pattern), copy.deepcopy(test_seqs[i])
                    del pattern[0]
                    pattern.append(test_seqs[i])
                if pos_marked is None: print('test_seqs', test_seqs, 'normal?', trueLabel==NORMAL)
                if not bSkip and pos_marked is not None and  trueLabel==ABNORMAL and isDetectedNormal(maxPrevProb, minPredProb, nLogSeqs, trueLabel, -1): # -1 means final/duplicated check
                    FN_byLog.append(nLogSeqs)

                if bWARN:
                    nAnomalies += 1
                    toWrite_detecRsts_seq.append(seq[-1].split()[-1]+":Anomaly"+":trueLabel:%d"%(trueLabel)+"\n")
                else:
                    toWrite_detecRsts_seq.append(seq[-1].split()[-1]+":Normal"+":trueLabel:%d"%(trueLabel)+"\n")

                fp_seq_rsts.write(' '.join(toWrite_detecRsts_seq))

            summ = '------summary---- #anomalies: %d, total: %d, rate: %s \n'%(nAnomalies, nLogSeqs, str(1.0*nAnomalies/nLogSeqs))
            summ += '#true normal: %d, true abnormal: %d \n'%(nNORMAL, nABNORMAL)
            summ += 'fpCnt: %d, unique: %d, FP log seq content: %s \n'%(len(FP_byLog), len(set(FP_byLog)), str(FP_byLog))
            summ += 'fnCnt: %d, unique: %d, FN log seq content: %s \n'%(len(FN_byLog), len(set(FN_byLog)), str(FN_byLog))
            fp_seq_rsts.write(summ)
            print(summ)
            print('FPpreds len', len(FPpreds), FPpreds)


saver = tf.train.Saver(tf.all_variables(), reshape=True,  max_to_keep=None)


#modelToEval = ['epoch99_validLoss0.212_validAcc91.67.histLen10.bTopKFalse_topPerc0.999_lowerPerc0.99_lossBnd0_12fn_44fp']
#modelToEval = ['epoch99_validLoss0.212_validAcc91.67.histLen10.bTopKFalse_topPerc0.999_lowerPerc0.99_lossBnd0_40fn_43fp']
#modelToEval = ['non_dp/epoch100_validLoss0.236_validAcc91.11_eps0_delta1e-05_sigma2.0_clip1.0', 'dp_clip1.0_delta1e-05/epoch100_validLoss0.507_validAcc88.11_eps0.9585177585005272_delta1e-05_sigma1.0_clip1.0'] #, 'dp_clip1.0_delta1e-05/epoch100_validLoss0.515_validAcc87.20_eps0.2495376875746713_delta1e-05_sigma2.0_clip1.0']
#modelToEval = ['non_dp/epoch100_validLoss0.236_validAcc91.11_eps0_delta1e-05_sigma2.0_clip1.0']
#modelToEval = ['dp_clip1.0_delta1e-05/epoch100_validLoss0.507_validAcc88.11_eps0.9585177585005272_delta1e-05_sigma1.0_clip1.0'] #, 'dp_clip1.0_delta1e-05/epoch100_validLoss0.515_validAcc87.20_eps0.2495376875746713_delta1e-05_sigma2.0_clip1.0']
modelToEval = ['dp_clip1.0_delta1e-05/epoch100_validLoss0.515_validAcc87.20_eps0.2495376875746713_delta1e-05_sigma2.0_clip1.0']
if logType=='android': modelToEval = ['epoch462_validLoss2.252_validAcc33.35'] #['hist10/epoch1000_validLoss2.306_validAcc32.81'] #['epoch52_validLoss2.392_validAcc31.59']
#modelToEval = ['epoch99_validLoss0.212_validAcc91.67.histLen10.bTopKFalse_topPerc0.999_lowerPerc0.99_lossBnd0_14fn_4fp']


fds = [os.path.join(MODEL_FD, '%s'%(logType), name) for name in modelToEval]
if args.model is not None:
    fds = args.model.split()
    print('==============from args.model', fds)
fds = args.model.split()
for ldFile in fds:
    t1 = datetime.datetime.now()
    eval(ldFile)
    t2 = datetime.datetime.now()
    print('total time', t2-t1)
