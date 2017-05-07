# -*- coding: utf-8 -*-
"""
Ce programme entraine un réseau de neuronne pour detecter différent phoneme dans un fichier .wav
il supporte actuelle seulement le "A", "E" ,"I"

il faut avoir un dataset conséquent pour obtenir des résultat car le réseau a tendance calculer la somme des point de chaque
forme d'onde plus tot que d'en identifier la forme.

Il faut avoir des fichier 0E.wav, 1E.wav, 2E.wav,
                          0A.wav, 1A.wac, 2A.wav
                          0I.wav, 1I.wac, 2I.wav
                          etc, etc dans le dossier ou est executer ce script
                          

7 mai 2017
Nathann Morand, natmo@hotmail.ch
"""
import wave
import struct
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import keras.backend as K


print K.backend()
numpy.random.seed(3)

AnumberOfSample = 1
EnumberOfSample = 3
InumberOfSample = 1
DataAugmentation = 160
X = []

#Data augmentation
for i in range(0,AnumberOfSample): #A
    x0 = []
    waveFile = wave.open((str(i)+'A.wav'), 'r')
    #length = waveFile.getnframes()
    for j in range(0,160):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        x0.append(((data[0])+32768)/float(65636))
    X.append(x0)
    for j in range(0, (DataAugmentation-1)):
        X.append((x0[j:] + x0[:j]))
    waveFile.close()

for i in range(0,EnumberOfSample): #E
    x0 = []
    waveFile = wave.open((str(i)+'E.wav'), 'r')
    #length = waveFile.getnframes()
    for j in range(0,160):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        x0.append(((data[0])+32768)/float(65636))
    X.append(x0)
    for j in range(0, (DataAugmentation-1)):
        X.append((x0[j:] + x0[:j]))
    waveFile.close()
    
for i in range(0,InumberOfSample): #I
    x0 = []
    waveFile = wave.open((str(i)+'I.wav'), 'r')
    #length = waveFile.getnframes()
    for j in range(0,160):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        x0.append(((data[0])+32768)/float(65636))
    X.append(x0)
    for j in range(0, (DataAugmentation-1)):
        X.append((x0[j:] + x0[:j]))
    waveFile.close()

print(X)

Y = []
#each section genere the ouput for the A, E, I
for i in range(0,(AnumberOfSample*DataAugmentation)): #A
    Y.append([1,0,0])

for i in range(0,(EnumberOfSample*DataAugmentation)): #E
    Y.append([0,1,0])
    
for i in range(0,(InumberOfSample*DataAugmentation)): #I
    Y.append([0,0,1])
    
print(Y)

# create model
model = Sequential()
model.add(Dense(256, input_dim=160, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
print K.backend()
print(keras.__version__)
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=400)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Genere la donné de test
XPredict = []
for i in range(3,4): #Utilise 3E.wav
    x0 = []
    waveFile = wave.open((str(i)+'E.wav'), 'r')
    #length = waveFile.getnframes()
    for j in range(0,160):
        waveData = waveFile.readframes(1)
        data = struct.unpack("<h", waveData)
        x0.append(((data[0])+32768)/float(65636))
    XPredict.append(x0)
    for j in range(0, (DataAugmentation-1)):
        XPredict.append((x0[j:] + x0[:j]))
    waveFile.close()


predictions = model.predict(XPredict)
print(predictions)
