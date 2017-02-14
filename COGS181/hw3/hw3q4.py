# HW2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.interactive(True)

# Problem 4

# Organizing input data
data = open('Q4_data.txt','r') 
trainX = []
trainY = []
testX = []
testY = []
into_test_virginica = 0
into_test_versicolor = 0
for line in data:
    info = line.split(',')
    #print info
    features = info[:4]
    features = [float(f) for f in features]
    category = info[-1]
    #print category[-2]
    # category can either be Iris-virginica or Iris-versicolor
    # Iris-virginica is +1 and Iris versicolor is 0
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
trainX = np.transpose(trainX)
testX = np.matrix(testX)
testX = np.transpose(testX)
print trainX.shape
#print len(trainX), len(trainY), len(testX), len(testY)

# sigmoid function
# takes in W, b, X, returns vectors F
# W: num_neuron_next * num_neuron_cur
# b: num_neuron_next * 1
# X: num_neuron_cur * num_samples
def sigmoid(W, b, X):
    # F is the decimal value from sigmoid function       
    F = np.zeros((W.shape[0], X.shape[1]))
    W = np.matrix(W)
    X = np.matrix(X)
    for j in range(W.shape[0]):
        w = W[j,:]
        for i in range(X.shape[1]):
            x = X[:,i]
            #print w.shape, x.shape, b.shape
            #print w, x, b
            #print w*x
            f = 1.0 / (1 + np.exp(-w*x-b[j,:]))
            #print f
            F[j,i] = f
            #print f
    #print F
    return F

# error rate function
# takes in F from sigmoid and original label Y
# returns D (diff in label), e (percentage of error)
# F, Y: 1 * num_samples
def error_rate(F, Y):
    F = np.matrix(F)
    Y = np.matrix(Y)
    Y_hat = np.zeros(Y.shape)
    D = np.zeros(Y.shape)
    for j in range(F.shape[1]):
        if F[0,j]>=0.5:
            Y_hat[0,j] = 1
        else:
            Y_hat[0,j] = 0
        if Y_hat[0,j]!=Y[0,j]:
            D[0,j] = 1.0
    e = float(np.sum(D)) / Y.shape[1]
    return D, e

# gradient function
# takes in W1, b1, W2, b2, X, Y, returns value g
def gradient(W1, b1, W2, b2, X, Y):
    F1 = sigmoid(W1, b1, X)
    F2 = sigmoid(W2, b2, F1)
    # compute g_W1
    # compute F-Y
    diff = F2 - np.matrix(Y)
    #print 'diff\n', diff
    # compute element-wise multiplicatin of A=w2*sig1*(1-sig1)
    A = np.zeros(F1.shape)
    A[0,:] = W2[0,0] * F1[0,:]
    A[1,:] = W2[0,1] * F1[1,:]
    B = np.zeros(A.shape)
    for i in range(A.shape[1]):
        d = diff[0,i]
        B[:,i] = A[:,i] * d
    g_W1 = B * np.transpose(X)
    #print 'B\n', B
    #print 'X\n', np.transpose(X)
    #print 'g_w1\n', g_W1
    # compute g_b1
    g_b1 = A * np.transpose(diff)
    # compute g_w2
    #print 'diff\n', diff.shape, diff
    #print 'F1\n', F1.shape, np.transpose(F1)
    g_W2 = diff * np.transpose(F1)
    # compute g_b2
    g_b2 = np.sum(diff)
    #print 'g_w2\n', g_W2
    #print 'g_b2\n', g_b2
    return g_W1, g_b1, g_W2, g_b2

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

def loss(W1,b1,W2,b2,X,Y):
    F1 = sigmoid(W1, b1, X)
    F2 = sigmoid(W2, b2, F1)
    Y = np.matrix(Y)
    L = 0
    for i in range(Y.shape[1]):
        y = Y[0,i]
        f = F2[0,i]
        L -= y*np.log(f) + (1-y)*np.log(1-f)
    return L

#learning parameters W and b
# Initialize W and b to be zeros
W1 = np.random.random((2,4))
b1 = np.random.random((2,1))
W2 = np.random.random((1,2))
b2 = np.random.random((1,1))
all_pass = False
itr = len(trainX)
lr = 0.01
counting=0
#trainY = np.matrix(trainY)
#print 'trainy shape', trainY.shape
train_loss = []
while not all_pass and counting<1000:
    # compute train loss
    L = loss(W1,b1,W2,b2,trainX,trainY)
    train_loss.append(L)
    print 'loss', L
    counting += 1
    #print 'before updating in loop', W, b
    # check training data to see if it is classified correctly
    F1 = sigmoid(W1, b1, trainX)
    F2 = sigmoid(W2, b2, F1)
    D, e = error_rate(F2,trainY)
    #print ' and trainY', H, F, trainY
    print 'error', e
    '''
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
    '''
    if e==0:
        print 'yay'
        all_pass = True

    # if not correct, update parameters
    g_W1, g_b1, g_W2, g_b2 = gradient(W1, b1, W2, b2, trainX, trainY)
    #print L_w,L_b
    if not all_pass:
        W1 += -lr * g_W1
        b1 += -lr * g_b1
        W2 += -lr * g_W2
        b2 += -lr * g_b2
    print 'after updating in loop', W1, b1, W2, b2
    print '# itr: ', counting
    #if counting>1:
    #    break
#print W,b
print '\nresults of training'
print '\ntraining loss:\n', train_loss
print 'num training itr:', counting
#measures(F, trainY)
#print F
#print trainY
# Test parameters on testing data set
print '\n testing error:\n'
testY = np.matrix(testY)
F1 = sigmoid(W1, b1, testX)
F2 = sigmoid(W2, b2, F1)
D, e = error_rate(F2,testY)
#print ' and trainY', H, F, trainY
print 'error', e
#print H_test
#print F_test
#print testY

# plotting
xcoords = range(len(train_loss))
plt.plot(xcoords, train_loss)
axes = plt.gca()
#axes.set_xlim([0,6])
#axes.set_ylim([0,0.6])
plt.grid()
plt.title('Training loss')
plt.xlabel('Iterations')
plt.ylabel('Training loss')
plt.show()
plt.savefig('plot.png')




