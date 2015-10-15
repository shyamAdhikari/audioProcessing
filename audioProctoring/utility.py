'''
Created on Dec 2, 2013

@author: shyam
'''
import numpy as np
from scipy.fftpack import dct

def hz2mel(hz):
    ''' 
    function to convert frequency to mel scale.
    '''
    mel = 1125* np.log(1 + hz/float(700))
    return mel

def mel2hz(mel):
    ''' 
    function to convert mel frequency back to hertz .
    '''
    hz = 700 * (np.exp(mel/1125)-1)
    return hz

def melFilterBank(numFilter,numFFT,samplingFreq,lowerFreq,higherFreq):
    '''
    function to create mel spaced filter banks.
    '''
    mflh = np.array([lowerFreq, higherFreq])
    melh1 = hz2mel(mflh)
    m = np.linspace(melh1[0], melh1[1], numFilter+2)
    h = mel2hz(m)    
    f = ((numFFT+1)*h/float(samplingFreq))    
    H = np.zeros((numFilter,numFFT))
    
    for j in range(1,numFilter+1):
        for k in range(0,numFFT):
            if (k < round(f[j-1])):
                H[j-1,k] = 0
            elif (k >= round(f[j-1]))and (k <= round(f[j])):
                H[j-1,k] = (k - round(f[j-1]))/round(f[j]-f[j-1])
            elif (k > round(f[j]))and (k <= round(f[j+1])):
                H[j-1,k] = (-k + round(f[j+1]))/round(-f[j]+f[j+1])
            elif (k > round(f[j+1])):
                H[j-1,k] =0
    return H

def mfccFeature(frameSpectrum,melFilterBank,coeffs):
    
    '''
    Calculate MFCC features for a audio frame.
    Input:
        frameSpectrum : spectrum of frame
        melFilterBank : filterBank description
        coeffs: number of mel coefficients required
    Output:
        mfccFinal : mfcc coefficients
    '''
    
    size = melFilterBank.shape[0]    
    mfccSum = np.zeros((1,size))    
    for j in range(0,size):
        mfccSum[0,j]= np.dot (frameSpectrum , melFilterBank[j,:])
        
    mfccLog = np.log(mfccSum)
    mfccDCT = dct(mfccLog)
    n = np.arange(1,coeffs+1)
    L = 22
    lift = 1 + L/float(2)*np.sin(np.pi*n/float(L))    
    mfcc = mfccDCT[0,0:coeffs]*lift    
    #energy = np.sum(frameSpectrum)
    mfccFinal = mfcc[0:coeffs]
    #mfccFinal[0] = energy               
    return mfccFinal

def frameEnergy(frame):
    '''
    compute energy of an audio frame.
    '''
    tmp = np.sum(frame**2)
    energy = np.log(tmp)
    return energy

def deltaCoefficients(MF):
    '''
    compute delta coefficients from mel coefficients
    '''
    deltaCoeff = (MF [3,:]+MF[4,:]-MF [0,:]-MF[1,:])/10
    return deltaCoeff


        
    
    