import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg


def softmax(X):
    m, n = X.shape
    return np.exp(X) / np.sum(np.exp(X), axis = 1).reshape(-1,1)

class NN:
    def __init__(self, input_size, hidden_size, output_size, std = 1e-4):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * std
        self.params['b1'] = np.random.randn(hidden_size) * std
        self.params['W2'] = np.random.randn(hidden_size, output_size) * std
        self.params['b2'] = np.random.randn(output_size) * std
    
    def prop(self, X, Y, norp):
        if len(X.shape) != 2:
            raise Exception('Please feed X of 2D')
        m, n = X.shape
        W1, b1, W2, b2 = tuple(self.params.values())
        
        z1 = X.dot(W1) + b1
        h1 = np.maximum(0, z1)
        scores = h1.dot(W2) + b2
        #scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
        P = softmax(scores)
        correct_P = P[range(m), list(Y)]
        loss = -np.sum(np.log(correct_P)) / m
        loss += (np.sum(np.square(W1)) + np.sum(np.square(W2))) * norp
        #print(loss)
        
        grads = {}
        dscores = P
        dscores[range(m), list(Y)] -= 1
        dscores /= m
        dW2 = h1.T.dot(dscores)
        db2 = np.sum(dscores, axis = 0)
        dh1 = dscores.dot(W2.T)
        dh1[h1 <= 0] = 0
        dW1 = X.T.dot(dh1)
        db1 = np.sum(dh1, axis = 0)
        dW1 += norp * W1
        dW2 += norp * W2
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['b2'] = db2
        
        return loss, grads
    
    def train(self, X, Y, X_cv, Y_cv, alpha = 0.001, alpha_decayrate = 0.95, \
              norp = 0, f = 0.9, epochs = 50, batch_size = 200):
        
        def takebatch():
            batch_index = np.random.choice(m, batch_size)
            return X[batch_index, ...], Y[batch_index]
            
        m, n = X.shape
        m_cv, n_cv = X_cv.shape
        iterations = int(max(m / batch_size, 1) * epochs)
        v_W1, v_W2, v_b1, v_b2 = 0,0,0,0
        loss_history = []
        train_acc_history = []
        cv_acc_history = []
        epochs = 1
        
        for i in range(iterations):
            Xbatch, Ybatch = takebatch()
            loss, grads = self.prop(Xbatch, Ybatch, norp)
            loss_history.append(loss)
            
            v_W2 = f * v_W2 - alpha * grads['W2']
            v_W1 = f * v_W1 - alpha * grads['W1']
            v_b2 = f * v_b2 - alpha * grads['b2']
            v_b1 = f * v_b1 - alpha * grads['b1']
            self.params['W2'] += v_W2
            self.params['W1'] += v_W1
            self.params['b1'] += v_b1
            self.params['b2'] += v_b2
            
            if i >= epochs * m / batch_size:
                alpha *= alpha_decayrate
                epochs += 1
                train_acc = (self.predict(Xbatch) == Ybatch).mean()
                cv_acc = (self.predict(X_cv) == Y_cv).mean()
                train_acc_history.append(train_acc)
                cv_acc_history.append(cv_acc)
                print('train_acc: %f, cv_acc: %f in epocs %d'%(train_acc, cv_acc, epochs - 1))
            
        return {'loss_his':loss_history, 'train_acc_his':train_acc_history, 'cv_acc_his':cv_acc_history}
    
    def predict(self, X):
        W1, b1, W2, b2 = tuple(self.params.values())
        
        z1 = X.dot(W1) + b1
        h1 = np.maximum(0, z1)
        scores = h1.dot(W2) + b2
        
        y_pred = np.argmax(scores, axis = 1)
        return y_pred
    
def getData():
    root = r'./train/'
    classestype = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    data = pd.DataFrame()
    file_name = pd.Series(dtype = str)
    classes = pd.Series(dtype = str)
    for i in classestype:
        name = []
        classname = []
        for j in os.listdir(root + i + r'/'):
            name.append(j)
            classname.append(i)
        file_name = file_name.append(pd.Series(name), ignore_index=True)
        classes = classes.append(pd.Series(classname), ignore_index=True)
    data['file_name'] = file_name
    data['class'] = classes
    data = data.sample(frac = 1).reset_index(drop=True)
    (n,m) = data.shape
    traindata = np.ndarray((n, 48*48))
    for i in range(len(data['file_name'])):
        f = mpimg.imread(root+data['class'][i]+r'/'+data['file_name'][i])
        traindata[i]=f.flatten()
    test = pd.read_csv('submission.csv')
    (n, m) = test.shape
    testdata = np.ndarray((n, 48*48))
    root = r'./test/'
    for i in range(len(test['file_name'])):
        f = mpimg.imread(root+test['file_name'][i])
        testdata[i]=f.flatten()
    trainclass = np.array(data['class'].map(lambda x : 0 if x == 'angry' else (1 if x == 'disgust' else (2 if x == 'fear' \
                          else (3 if x == 'happy' else (4 if x == 'neutral' else (5 if x == 'sad' else 6)))))))
    return traindata, trainclass, testdata
    
def test(net, Xtest):
    testYnum = net.predict(Xtest)
    testdata = pd.read_csv('submissiono.csv')
    testdata['class'] = pd.Series(testYnum.flatten())
    testdata['class'] = np.array(testdata['class'].map(lambda x : 'angry' if x == 0 else ('disgust' if x == 1 else ('fear' if x == 2 \
                          else ('happy' if x == 3 else ('neutral' if x == 4 else ('sad' if x == 5 else 'surprise')))))))
    testdata.to_csv('submission.csv',index = False)

datatrainX, datatrainY, datatestX = getData()
np.save('datatrainX', datatrainX)
np.save('datatrainY', datatrainY)
np.save('datatestX', datatestX)
'''datatrainX = np.load('datatrainX.npy')
datatrainY = np.load('datatrainY.npy')
datatestX = np.load('datatestX.npy')'''
numtrain = 28000
numval = 709
numtest = 7178
Xtrain = datatrainX[:numtrain, ...]
Ytrain = datatrainY[:numtrain, ...]
Xval = datatrainX[numtrain:numtrain + numval, ...]
Yval = datatrainY[numtrain:numtrain + numval, ...]
Xtest = datatestX[:numtest, ...]
#Ytest = datatestY[:numtest]
meanx = np.mean(Xtrain, axis = 0)
Xtrain -= meanx
Xval -= meanx
Xtest -= meanx

input_size = 48*48
hidden_size = 75
output_size = 7
net = NN(input_size, hidden_size, output_size)
history = net.train(Xtrain, Ytrain, Xval, Yval)
val_acc = (net.predict(Xval) == Yval).mean()
train_acc = (net.predict(Xtrain) == Ytrain).mean()
print(val_acc)
print(train_acc)
test(net, Xtest)