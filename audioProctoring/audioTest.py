'''
Created on Nov 20, 2013

@author: shyam
'''


from scipy.io.wavfile import read
from scipy.signal import hann
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import numpy as np
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(__file__),'LIBSVM/python'))
from svmutil import *
import utility as util

def featuresToDict (sim):
    """
    Convert similarity vector to the form required by LibSVM.
    Input:
        sim : (double) list of similarity score
    Output:
        prob_x : (double) array in the form required by LibSVM.
    """    
    prob_x = []
    xi = {}
    for x in range(0,len(sim)):
        xi[int(x+1)] = float(sim[x])
    prob_x += [xi]
    return prob_x

def powerSpectrum(Frame,window,NFFT):
    '''
    Compute power spectrum of a frame. 
    '''
    winFrame = window * Frame
    ffty =   abs(fft(winFrame,NFFT))
    magy = 1/NFFT*ffty**2    
    return magy

def framePrediction(feature,model):
    '''
    Prediction of individual frame using SVM.
    '''
    x = featuresToDict(feature)
    y = [1]
    p_label,_,p_val = svm_predict(y,x,model,'-b 1')
    return (p_label,p_val)

def writetoFile(filename,label,feature):
    
    '''
    Write training data to file in format 
    required by LibSVM.
    '''
        
    #features = feature.tolist()
    f = open(filename,'a')
    f.write('%d\t' %label)
    
    for i in range (0,len(feature)):
        f.write('%d:%f\t'%(i+1,feature[i]))
    f.write('\n')
    f.close()
    
def KbandFeatures(K,sourceFeature,NFFT):
    bandFeature = []
    for k in range(0,K):
        init = round(NFFT*k/float(2*K))
        iend = round(NFFT*(k+1)/float(2*K))-1
        tmp = np.sum(sourceFeature[init:iend])
        bandFeature.append(tmp)
    return bandFeature
    
def LTSEfeature(audioFileName,trainingDataFile,label):
    
    '''
    Long term spectral envelope feature calculation.
    '''
    
    fs,audio = read (audioFileName)
    audio = audio/float(4000)        
    # frame duration in ms
    frame_length = 20
    # overlap duration in ms
    frame_overlap = 10
    N = len (audio)
    nsample = round(frame_length*fs/1000)
    noverlap = round(frame_overlap*fs/1000)
    # FFT length
    NFFT = 2*nsample
    # Hanning window
    window = hann(nsample)
    offset = nsample-noverlap
    max_m = round((N-NFFT)/offset)
    nFrame = 8
    init = round(nFrame/2+1)
    max_m = max_m-init
    
    SE = np.zeros((nFrame,NFFT))
    LTSE = np.zeros((max_m,NFFT))
    flag = 0   
    
    for m in range(0,int(max_m-init)):
        begin = m*offset 
        iend = m*offset + nsample
        Frame = audio[begin:iend]
        magy = powerSpectrum(Frame,window,NFFT)
        SE[m%nFrame,:] = magy
    
        if m%nFrame == nFrame-1 or flag == 1:
            LTSE[m-init,:] = SE.max(0)
            bandFeature = KbandFeatures(16,LTSE[m-init,:],NFFT)
            
            if (trainingDataFile):
                writetoFile(trainingDataFile,label,bandFeature)
            #LTSE[m-init,:] = LTSE[m-init,:]/LTSE[m-init,:].max()
            #writetoFile(trainingDataFile,label,LTSE[m-init,:])             
            flag = 1
            
def noiseEstimate(signal,fs,nFrame,frame_length,frame_overlap):
    '''
    Estimation of noise from the initial nFrames.
    '''
    
    nsample = round(frame_length*fs/1000)
    noverlap = round(frame_overlap*fs/1000)
    # FFT length
    NFFT = 2*nsample
    offset = nsample-noverlap
    window = hann(nsample)
    
    nsum = np.zeros((1,NFFT))    
    
    for m in range(0,nFrame):
        begin = m*offset 
        iend = m*offset + nsample
        Frame = signal[begin:iend]
        magy = powerSpectrum(Frame,window,NFFT)
        nsum =  nsum + magy
    
    noisePow = nsum/float(nFrame)    
    return noisePow     
         
def ltsdVAD(LTSD,threshold):
    VAD = []
    
    for j in range(0,len(LTSD)):
        if LTSD[j]> threshold:
            VAD.append(1)
        else:
            VAD.append(0)
    return VAD

def zeroCrossingRate(frame,window):
    frame = frame * window
    
    frame_1 = frame[0:len(frame)-1]
    frame_2 = frame [1:]
    zcr = np.sum(np.absolute(np.sign(frame_2)-np.sign(frame_1)))/float(len(frame))
    return zcr

def Cepstrum(frame,window):
    winFrame = frame * window
    NFFT = len(frame)
    fft_sig = abs(fft(winFrame,NFFT))
    cep =  ifft(np.log(fft_sig))   
    return cep

def pitchTrack(cep,fs):
    ms2 =  (math.floor(fs*2/1000.0))
    ms20 =  (math.floor(fs*20/1000.0))
    cep1 =cep[ms2:ms20]
    idx = np.argmax(abs(cep1))
    f0 = fs/(ms2+idx)       
    return f0/float(500)

def fundamentalFreq(frame,window,fs):
    '''
    Fundamental frequency detection
    '''
    cep = Cepstrum(frame,window)
    f0 = pitchTrack(cep,fs)
    return f0

def preEmphasis(frame,alpha):
    '''
    pre-emphasis filtering
    '''
    
    emp_frame = np.zeros((len(frame)))
    
    for j in range(0,len(frame)):
        if j >= 1:
            emp_frame[j] = frame[j]-alpha*frame[j-1]
    return emp_frame    
    

def MFCCfeatureExtraction(audioFileName,trainingDataFile,label,DEBUG):
    '''    
    MFCC feature extraction for training data generation.
    '''
    
    #,trainingDataFile,label
    fs,audio = read (audioFileName)
    audio = audio/float(4000)        
    # frame duration in ms
    frame_length = 25
    # overlap duration in ms
    frame_overlap = 10
    N = len (audio)
    nsample = round(frame_length*fs/1000)
    noverlap = round(frame_overlap*fs/1000)
    # FFT length
    NFFT = 2*nsample
    # Hanning window
    window = hann(nsample)
    offset = nsample-noverlap
    max_m = round((N-NFFT)/offset)    
    
    numFilter = 26
    fl = 0*fs
    fh = 0.5*fs
    melFilBank  = util.melFilterBank(numFilter, int(NFFT/2), fs, fl, fh)
    coeffs = 13 
    N1 = 5
    flag = 0
    count = 0
    flag2 = 0    
    
    frames = int(max_m)
    if (DEBUG):        
        mfccTrack = np.zeros((coeffs,frames))
        deltaTrack = np.zeros((coeffs,frames))
        delta2Track = np.zeros((coeffs,frames))
    
    mQ = []
    dQ = []
    ZCR = []
    F0 = []    
    
    for m in range(0,frames):
        begin = m*offset 
        iend = m*offset + nsample
        Frame = audio[begin:iend]
        magy = powerSpectrum(Frame,window,NFFT)
        mfccfeature = util.mfccFeature(magy[0:int(NFFT/2)], melFilBank, coeffs)
        mfccfeature = mfccfeature/np.absolute(mfccfeature).max(0)
        mQ.append(mfccfeature)
        
        zcr_temp = zeroCrossingRate(Frame,window)        
        f0 = fundamentalFreq(Frame,window,fs)
        ZCR.append(zcr_temp)
        F0.append(f0)
        
        if (DEBUG) :
            mfccTrack[0:coeffs,m] = mfccfeature[0:coeffs]  
        
        if m%N1 == N1-1 or flag == 1:
            
            MD = util.deltaCoefficients(np.asarray(mQ))
            MD = MD/np.absolute(MD).max(0)
            dQ.append(MD)
            mQ.pop(0)
            ZCR.pop(0)
            F0.pop(0)            
            
            flag = 1
             
            if (DEBUG) : 
                deltaTrack[0:coeffs,m] = MD[0:coeffs]           
                        
            if count%N1 == N1-1 or flag2 == 1:
                
                MDD = util.deltaCoefficients(np.asarray(dQ))
                MDD = MDD/np.absolute(MDD).max(0)
                dQ.pop(0)
                               
                feature =[mQ[0],dQ[2],MDD]                
                feature1 = np.reshape(np.vstack(feature),3*coeffs)
                feature2 = feature1.tolist()
                feature2.append(ZCR[0])
                feature2.append(F0[0])
                feature2.append(ZCR[0]-ZCR[1])
                feature2.append(F0[0]-F0[1])   
                if (trainingDataFile):                                  
                    writetoFile(trainingDataFile,label,feature2)
                flag2 = 1
                
                if(DEBUG):
                    delta2Track[0:coeffs,m] = MDD[0:coeffs]
            count = count+1
            
    if (DEBUG):  
        T = np.arange(round(nsample/2),N-1-round(nsample/2),(nsample-noverlap))/fs;
        L1 = T [0:int(max_m)];   
        t = np.linspace(0,N,N)/fs;
        
        plt.subplot(4,1,1)
        plt.plot(t,audio)    
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(audioFileName)
        plt.xlim([0, T[-1]])
        
        plt.subplot(4,1,2)
        plt.plot(L1,mfccTrack[0,:],color='r')
        plt.plot(L1,mfccTrack[1,:],color='g')
        plt.plot(L1,mfccTrack[2,:],color='b')
        plt.xlim([0, T[-1]])
        
        plt.subplot(4,1,3)
        plt.plot(L1,deltaTrack[0,:],color='r')
        plt.plot(L1,deltaTrack[1,:],color='g')
        plt.plot(L1,deltaTrack[2,:],color='b')
        plt.xlim([0, T[-1]])
            
        plt.subplot(4,1,4)
        plt.plot(L1,delta2Track[0,:],color='r')
        plt.plot(L1,delta2Track[1,:],color='g')
        plt.plot(L1,delta2Track[2,:],color='b')
        plt.xlim([0, T[-1]])
        plt.show()
        
def MFCC_SVM(audioFileName,model,DEBUG):
    '''
    Calculate MFCC features from audio frames and use
    trained SVM for frame prediction:
    1. divide audio file into overlapping frames 
    2. window the frame
    3. compute MFCC features for each frame
    4. predict using SVM trained on MFCC features
    
    Input:
        audioFileName : input test audio file
        model : SVM model
        DEBUG : debug flag
    Output:
        label : predicted label for all frames of the audio
        tag :  probability of prediction    
     
    '''
    
    #,trainingDataFile,label
    fs,audio = read (audioFileName)
    audio = audio/float(4000)        
    # frame duration in ms
    frame_length = 25
    # overlap duration in ms
    frame_overlap = 10
    N = len (audio)
    nsample = round(frame_length*fs/1000)
    noverlap = round(frame_overlap*fs/1000)
    # FFT length
    NFFT = 2*nsample
    # Hanning window
    window = hann(nsample)
    offset = nsample-noverlap
    max_m = round((N-NFFT)/offset)    
    
    numFilter = 26
    fl = 0*fs
    fh = 0.5*fs
    melFilBank  = util.melFilterBank(numFilter, int(NFFT/2), fs, fl, fh)
    coeffs = 13 
    N1 = 5
    flag = 0
    count = 0
    flag2 = 0    
    
    frames = int(max_m)
      
    mQ = []
    dQ = []
    tag = [] 
    ZCR = []
    F0 = []    
    label = []
    alpha = 0 # pre-emphasis factor
    
    for m in range(0,frames):
        begin = m*offset 
        iend = m*offset + nsample
        frame = audio[begin:iend]
        Frame = preEmphasis(frame,alpha)
        magy = powerSpectrum(Frame,window,NFFT)
        mfccfeature = util.mfccFeature(magy[0:int(NFFT/2)], melFilBank, coeffs)
        mfccfeature =mfccfeature/np.absolute(mfccfeature).max(0)
        mQ.append(mfccfeature)
        
        zcr_temp = zeroCrossingRate(Frame,window)        
        f0 = fundamentalFreq(Frame,window,fs)
        ZCR.append(zcr_temp)
        F0.append(f0)        
        
        if m%N1 == N1-1 or flag == 1:
            
            MD = util.deltaCoefficients(np.asarray(mQ))
            MD = MD/np.absolute(MD).max(0)
            dQ.append(MD)
            mQ.pop(0)
            ZCR.pop(0)
            F0.pop(0)            
            flag = 1             
                                    
            if count%N1 == N1-1 or flag2 == 1:
                
                MDD = util.deltaCoefficients(np.asarray(dQ))
                MDD = MDD/np.absolute(MDD).max(0)
                dQ.pop(0)
                               
                feature =[mQ[0],dQ[2],MDD]                
                feature1 = np.reshape(np.vstack(feature),3*coeffs)
                feature2 = feature1.tolist()
                feature2.append(ZCR[0])
                feature2.append(F0[0])
                feature2.append(ZCR[0]-ZCR[1])
                feature2.append(F0[0]-F0[1])                  
                p_label,p_val = framePrediction(feature2,model)
                tag.append(p_val[0][0])
                label.append(p_label[0]>0) 
                  
                flag2 = 1           
               
            count = count+1
    if (DEBUG):  
        T = np.arange(round(nsample/2),N-1-round(nsample/2),(nsample-noverlap))/fs;
        #L1 = T [0:int(max_m)];   
        t = np.linspace(0,N,N)/fs;
        L2 = T [8:len(tag)+8];      
        
        plt.subplot(3,1,1)
        plt.plot(t,audio)    
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(audioFileName)
        plt.xlim([0, T[-1]])
        
        plt.subplot(3,1,2)
        plt.plot(L2,tag,color = 'r')
        plt.xlabel("Time")
        plt.ylabel("SVM-tag")
        plt.ylim([-1.2,1.2])
        plt.xlim([0, T[-1]])        
        
        
        plt.subplot(3,1,3)
        plt.plot(L2,label,color = 'g')
        plt.xlabel("Time")
        plt.ylabel("SVM-tag")
        plt.ylim([-1.2,1.2])
        plt.xlim([0, T[-1]])        
        plt.show()
        
    return label,tag
        

def LTSE_VAD(audioFileName):
    '''
    Voice activity detection with estimation of long-term spectral envelope.
    Adapted from :
    http://sirio.ugr.es/segura/pdfdocs/ES030097.PDF
        
    '''
    fs,audio = read (audioFileName)
    audio = audio/float(4000)
       
    # audio length in seconds
    N1 = len(audio)/fs    
    # frame duration in ms
    frame_length = 25
    # overlap duration in ms
    frame_overlap = 10
    N = len (audio)
    nsample = round(frame_length*fs/1000)
    noverlap = round(frame_overlap*fs/1000)
    # FFT length
    NFFT = 2*nsample
    # Hanning window
    window = hann(nsample)
    offset = nsample-noverlap
    max_m = round((N-NFFT)/offset)
    nFrame = 8
    init = round(nFrame/2+1)
    max_m = max_m-init
    
    SE = np.zeros((nFrame,NFFT))
    LTSE = np.zeros((max_m,NFFT))
    flag = 0
    tag = []
    LTSD = []
    zcr = [] 
    F0 = []   
    noiseFrame = 20    
    noisePow = noiseEstimate(audio,fs,noiseFrame,frame_length,frame_overlap)
    noiseUpdateCount = noiseFrame
    
    for m in range(0,int(max_m-init)):
        begin = m*offset 
        iend = m*offset + nsample
        Frame = audio[begin:iend]                
        magy = powerSpectrum(Frame,window,NFFT)
        SE[m%nFrame,:] = magy
    
        if m%nFrame == nFrame-1 or flag == 1:
            LTSE[m-init,:] = SE.max(0)
            tmp = 10 * np.log10(1/float(NFFT)*np.sum(LTSE[m-init,:]/noisePow))
            LTSD.append(tmp)
            
            if tmp <= 2:
                noisePow = noisePow + (-noisePow + LTSE[m-init,:] )/noiseUpdateCount
                noiseUpdateCount = noiseUpdateCount + 1.            
            
            #LTSE[m-init,:] = LTSE[m-init,:]/LTSE[m-init,:].max(0)        
            #p_label,_ = framePrediction(LTSE[m-init,:])        
            #tag.append(p_label)  
            flag = 1        
    #print tag
    VAD = ltsdVAD(LTSD,7)    
    L = len (VAD);
    T = np.arange(round(nsample/2),N-1-round(nsample/2),(nsample-noverlap))/fs;
    
    L1 = T [init:L+init];    
    t = np.linspace(0,N,N)/fs;
    plt.subplot(2,1,1)
    plt.plot(t,audio)    
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(audioFileName)
    plt.xlim([0, T[-1]])   
    
    
    plt.subplot(2,1,2)
    plt.plot(L1,VAD)
    plt.xlabel("Time")
    plt.ylabel("VAD-tag")
    plt.ylim([-0.2,1.2])
    plt.xlim([0, T[-1]])
    plt.show() 
    

def evaluatePerformance(testFile,modelName):
    '''
    Evaluate performance of the speech classification algorithm.
    Input :
        testFile : list of voice files
        modelName : SVM model for classification
    Output:
        hitrate: predicted true positives/total positives
    '''
    
    svm_model = svm_load_model(modelName) 
    voiceFile = [line.strip() for line in open(testFile)]
    total_frames = 0
    total_pos = 0
    
    for j in range(len(voiceFile)):
        
        File = voiceFile[j]
        label,_ = MFCC_SVM(File,svm_model,0)
        total_frames = total_frames + len(label)
        total_pos = total_pos + sum(label)
    
    hitrate = total_pos / float(total_frames)*100.
    
    return hitrate,total_frames 
            
