import os
import math
import ntpath


opath = input('>>>output path:')
sampleData = input('>>>sampleData: ')
sampleType = input('>>>sampleType: ')
n_sample =int( input('>>>number of sample: '))

dataName = ntpath.basename(sampleData)

n_input  =int( input('>>>number of input : '))
n_output =int( input('>>>number of output: '))

trainSet = int(n_sample/10)

trainType = ['sequence','sortRandomSet']
hiddenLayer = [1,2]
hiddenNode = [math.ceil(math.sqrt(n_input * n_output)) , int(n_input + n_output), math.ceil((n_input+n_output)/2)]

for trainTypeNow in trainType:
    for hiddenLayerNow in hiddenLayer:
        for hiddenNodeNow in hiddenNode:
            print(hiddenNodeNow)
            name = ''.join(opath+dataName+'T')
            name += str(trainType.index(trainTypeNow))
            name += "L"
            name += str(hiddenLayer.index(hiddenLayerNow))
            name += "N"
            name += str(hiddenNode.index(hiddenNodeNow) )
            f = open(name +'.nn','w')
            f.write('#########################################'+'\n')
            f.write('#Data         :'+sampleData.rjust(25)+'#\n')
            f.write('#Type         :'+sampleType.rjust(25)+'#\n')
            f.write('#Sample size  :'+str(n_sample).rjust(25)+'#\n')
            f.write('#Input        :'+str(n_input).rjust(25)+'#\n')
            f.write('#Output       :'+str(n_output).rjust(25)+'#\n')
            f.write('#set size     :'+str(trainSet).rjust(25)+'#\n')
            f.write('#########################################'+'\n')
            f.write('costFunction=mse'+'\n')
            f.write('#input_sample_setting'+'\n')
            f.write('sampleType='+sampleType+'\n')
            if sampleType == 'timeseries':
                f.write('trainFeature=5'+'\n')
            f.write('sampleData="'+sampleData+'"\n')
            if trainTypeNow == 'sequence':
                f.write('trainType='+trainTypeNow+'\n')
            elif trainTypeNow == 'sortRandomSet':
                f.write('trainType='+trainTypeNow+','+str(trainSet)+'\n')
            f.write('testType=sequence'+'\n')
            f.write('testStep=0'+'\n')
            f.write('#train_parameter'+'\n')
            if trainTypeNow == 'sortRandomSet':
                f.write('iteration=100000'+'\n')
            else:
                f.write('iteration=10000'+'\n')
            #f.write('stopTrainingCost=0.000001'+'\n')
            f.write('learningRate=0.8'+'\n')
            f.write('#neural_network_setting'+'\n')
            if hiddenLayerNow == 1:
                f.write('hidden=1,'+str(hiddenNodeNow)+'\n')
            elif hiddenLayerNow == 2:
                f.write('hidden=1,'+ str(math.ceil(int(hiddenNodeNow) * (float(n_input)/float(n_input+n_output))))+'\n')
                f.write('hidden=2,'+ str(math.ceil(int(hiddenNodeNow) * (float(n_output)/float(n_input+n_output))))+'\n')
            if sampleType == 'timeseries':
                f.write('output=1,tanh'+'\n')
            else:
                f.write('output=1'+'\n')
            f.close()

