'''
Created on Nov 21, 2013

@author: shyam
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'LIBSVM/python'))
from svmutil import *
import audioTest as aT


def SVMTraining(training_datafile,model_name,param):
    """ SVM Training"""
    y,x = svm_read_problem(training_datafile)
    prob = svm_problem(y,x)
    
    """parameters: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    # -t kernel type: (0: linear; 1: polynomial; 2: radial basis <default>; 3:sigmoid; 4: precomputed)
    # -s svm_type :(0: C-SVC; 1: nu-SVC; 2: one class SVM; 3: epsilon-SVR; 4: nu-SVR)
    # -c :(cost)
    # -g : gamma
    #param=svm_parameter('-t 2 -c 0.5 -g 0.1 -w1 20 -b 0')"""    
      
    m = svm_train(prob,param)
    svm_save_model(model_name,m)


def trainingData(voiceFileList,nonvoiceFileList,training_datafile):
    
    '''
    Prepare training data using voice and non-voice audio file list
    '''
    
    voiceFile = [line.strip() for line in open(voiceFileList)] 
    nonvoiceFile = [line.strip() for line in open(nonvoiceFileList)]
    
    for j in range(0,len(voiceFile)):
        vfl = voiceFile[j]
        print vfl
        label = 1
        #aT.LTSEfeature(vfl,training_datafile,label)
        aT.MFCCfeatureExtraction(vfl,training_datafile,label,0)
        
    for j in range(0,len(nonvoiceFile)):
        nvfl = nonvoiceFile[j]
        print nvfl
        label = -1
        aT.MFCCfeatureExtraction(nvfl,training_datafile,label,0)
        #aT.LTSEfeature(nvfl,training_datafile,label)
        
    pass

if __name__=="__main__":    
    
    voiceFileList = "E:/voice_data/voice_sample_training/adapted_voice/adapted_voice_train.txt"
    nonvoiceFileList = 'E:/voice_data/voice_sample_training/non_voice/train/noise_train.txt'   
       
    model_name = "./data/VAD_mfcc_v3_adapted1.model"  
    training_datafile = "./data/VAD_mfcc_v3_adapted.txt"    
    trainingData(voiceFileList,nonvoiceFileList,training_datafile)          
    param = svm_parameter('-t 2 -h 0 -c 1 -g .125 -b 1') 
    print ('training started...'+'\n')      
    SVMTraining(training_datafile,model_name,param)
        
         
   
    
                  
    