neural network practice

 nn file format
#############################################################
#nn file start
#ignore any text with '#' start
   costFunction=      //mse,nmse

#sample
#only classification is tested
#gradient check not finished 
#classification will set output nodes ( 0 ~ label max )
   smapleType=        //classification regression timeseries
   sampleData="path"  //path for training sample
   samplingType=      //all
                      //number,start,end
                      //bunch,data set size,number of data set
#train
   iteration=int
   learningRate=float

#test
	testType=         //all
    testStep=         //show the test detail per step

#neural network layer
#activation sigmoid tanh
#activation empty = sigmoid

   hidden=1,numberOfNode,activation
        .
        .
        .
    hidden=n,numberOfNode,activation
    output=n+1,2
#nnfile end
#############################################################

 sample file format
label0,label1,...,labelN:feature0,feature1,.....,featureN //label can set float number

