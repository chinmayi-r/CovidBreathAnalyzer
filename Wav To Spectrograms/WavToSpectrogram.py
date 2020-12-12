#This is a Google Cloud Function (serverless) that converts the wav file dropped in a Google bucket into a Spectrogram image file

from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib import stride_tricks
from librosa.core import amplitude_to_db
import wave
import os
from os import environ
from google.cloud import storage

client = storage.Client()

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (so that samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    print('stft() done')

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
    print('logscale_spec() done')
    return newspec, freqs

def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    print('_wav2array() done')
    return result

def readwav(myfile):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    # Get the file that has been uploaded to GCS
    bucket = client.get_bucket(myfile['bucket'])
    blob = bucket.get_blob(myfile['name'])
    
    try:
        f = open("/tmp/"+ myfile['name'],'wb')
        wav = blob.download_as_string()   
        blob.download_to_file(f)
        f.close()
        wav = wave.open(str("/tmp/"+ myfile['name']))
    
        rate = wav.getframerate()
    
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        nframes = wav.getnframes()
        data = wav.readframes(nframes)
        wav.close()
        array = _wav2array(nchannels, sampwidth, data)
    except Exception as e:
        print("Exception in rest of readwav")
        print(e)
   
    print('readwavfile() done')
    return rate, sampwidth, array

""" plot spectrogram"""
def plotstft(myfile, plotpath, save = True, binsize=2**10, colormap="jet"):
    print('in plotstft')
    
    fileName = myfile['name']
    fileNameWOExt = myfile['name']
    filenameWOExt = fileNameWOExt[0:-4]
    pngfileName = filenameWOExt+'.png'
    tmpplotpath = '/tmp/' + pngfileName
    plotpath = myfile['bucket']+'/'+pngfileName
 
    print("tmpplotpath: ", tmpplotpath)
    print("plotpath: ", plotpath)
    
    samplerate, sampwidth, samples = readwav(myfile)
    
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    #ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims = amplitude_to_db(np.abs(sshow))

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(20, 10))
    fig = plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.axis('off')
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    print('pltstft() forming graph done')

    if save:
        try:
            plt.savefig(tmpplotpath, bbox_inches="tight")
            bucket = client.get_bucket(myfile['bucket'])
            blob2 = bucket.blob(pngfileName)
            blob2.upload_from_filename(filename=tmpplotpath)
        except Exception as e:
            print("Exception in plotstft in savefig")
            print( e)
    else:
        plt.show()
    plt.close()
    
    print('pltstft() saving graph done')
    return plotpath

def hello_gcs(event, context, allowed_exts=("wav",)):

    myfile = event
    fileName = myfile['name']
    print(f"Uploaded file: {myfile['name']} of bucket: {myfile['bucket']}.")

    # Check that the file meets requirements
    if fileName.endswith(allowed_exts) :
        print(f" bucketName: {myfile['bucket']}")
        print(f" fileName: {myfile['name']}")
        
        print('Created: {}'.format(myfile['timeCreated'])) #this here for illustration purposes
        print('Updated: {}'.format(myfile['updated']))

        try:
            ims = plotstft(myfile,'')
        except Exception as e:
            print("Exception in main function")
            print( e)
            
        print(f"Converted file: {myfile['name']} of bucket: {myfile['bucket']} to spectrogram")
    else :
        print (f"Only files with a type of {allowed_exts} allowed. Exiting without processing {fileName}")
