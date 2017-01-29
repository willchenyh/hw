# HW2
import matplotlib.pyplot as plt
import numpy as np

# Problem 2 - Logistic Regression

# Organizing input data
data2 = open('Q2_data.txt','r') 
trainX = []
trainY = []
testX = []
testY = []
into_test_virginica = 0
into_test_versicolor = 0
for line in data2:
    info = line.split(',')
    #print info
    features = info[:4]
    features = [float(f) for f in features]
    category = info[-1]
    #print category[-2]
    # category can either be Iris-virginica or Iris-versicolor
    # Iris-verginica is +1 and Iris versicolor is -1
    if category[6]=='i':
        y = 1
        if into_test_virginica<15:
            testX.append(features)
            testY.append(y)
            into_test_virginica += 1        
        else:                                                                           
            trainX.append(features)
            trainY.append(y)
    else:
        y = 0
        if into_test_versicolor<15:
            testX.append(features)
            testY.append(y)
            into_test_versicolor += 1 
        else:
            trainX.append(features)
            trainY.append(y)
trainX = np.matrix(trainX)
testX = np.matrix(testX)
#print trainX, trainY
#print len(trainX), len(trainY), len(testX), len(testY)

# sigmoid function
# takes in W, X, b, returns vectors H, F
def confidence(W, X, b):
    # H is the decimal value from sigmoid function
    # F is the label 0 and 1            
    H = []
    F = []
    for i in range(len(X)):
        x = X[i,:]
        h = float(float(1) / (1 + np.exp(-x*W-b)))
        if h>=0.5:
            f = 1
        else:
            f = 0
        H.append(h)
        F.append(f)
    #print H, F
    return H, F

# cost function
# takes in y, W, X, b, returns value L
def cost(Y, P, X, b):
    L_w = 0
    diff = np.array(P) - np.array(Y)
    print 'diff',diff
    L_b = np.sum(diff)
    for i in range(len(X)):
        x = X[i, :]
        d = diff[i]
        l_w = d * x
        L_w += l_w
    return np.transpose(L_w), L_b

#print cost(trainY, W, trainX, b)
#H,F = confidence(W, trainX, b)
#print F[0]

def measures(F,Y):
    #print F
    F = np.array(F)
    Y = np.array(Y)
    c1 = 1
    c2 = 0
    tpos = 0.0
    fpos = 0.0
    tneg = 0.0
    fneg = 0.0
    for i in range(len(Y)):
        f = F[i]
        y = Y[i]
        #print f, y, c1, c2
        if f==c1 and y==c1:
            tpos += 1
        elif f==c1 and y==c2:
            fpos += 1
        elif f==c2 and y==c2:
            tneg += 1
        else:
            fneg += 1
    #print tpos, fpos, tneg, fneg
    accuracy = float(tpos + tneg) / len(Y)
    precision = tpos / float(tpos + fpos)
    recall = tpos / float(tpos + fneg)
    fvalue = 2*precision*recall / float(precision+recall)
    print 'accuracy', accuracy
    print 'precision', precision
    print 'recall', recall
    print 'fvale', fvalue


#learning parameters W and b
# Initialize W and b to be zeros
W = np.zeros((4,1))
b = 0
all_pass = False
itr = len(trainX)
lr = 0.001
counting=0
while not all_pass and counting<100000:
    counting += 1
    #print 'before updating in loop', W, b
    # check training data to see if it is classified correctly
    H, F = confidence(W, trainX, b)
    print 'H, F and trainY', H, F, trainY
    diff = np.array(trainY) - np.array(F)
    error = np.sum(diff)
    print 'error',error
    for i in range(itr):
        #if i%20==0:
        #   print i
        label = trainY[i]
        f = F[i]
        #print label, f
        if i==itr-1 and label==f:
            print 'yay'
            all_pass = True
        elif f!=label:
            break
    # if not correct, update parameters
    L_w, L_b = cost(trainY, H, trainX, b)
    #print L_w,L_b
    if not all_pass:
        W += -lr * L_w
        b += -lr * L_b
    print 'after updating in loop', W, b
    print '# itr: ', counting
#print W,b
print '\nresults of training'
measures(F, trainY)
#print F
#print trainY

# Test parameters on testing data set
H_test, F_test = confidence(W, testX, b)
print '\nresults of testing'
#print H_test
#print F_test
#print testY
measures(F_test, testY)






