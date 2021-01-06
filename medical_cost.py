import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def getData():
    path = 'train.csv'
    with open(path) as f:
        data = pd.read_csv(f)
    data['sex']=data['sex'].map(lambda x:1 if x == 'female' else 0)
    data['smoker']=data['smoker'].map(lambda x : 1 if x == 'yes' else 0)
    data.insert(len(data.columns)-1,'northeast', data['region'].map(lambda x : 1 if x == 'northeast' else 0))
    data.insert(len(data.columns)-1,'northwest', data['region'].map(lambda x : 1 if x == 'northwest' else 0))
    data.insert(len(data.columns)-1,'southeast', data['region'].map(lambda x : 1 if x == 'southeast' else 0))
    data.insert(len(data.columns)-1,'southwest', data['region'].map(lambda x : 1 if x == 'southwest' else 0))
    need = ['age','sex','bmi','children','smoker', 'northeast', 'northwest', 'southeast', 'southwest']
    for i in need:
        for j in need:
            for k in need:
                data.insert(len(data.columns)-1,i+j+k, data[i].mul(data[j]).mul(data[k]))
    data = data.drop(columns=['region'])
    #print(data)
    data = np.array(data, dtype = np.float64)
    return data

def computeCost(w, x, y):
    (n, m) = x.shape
    f = np.dot(x, w)
    tem = np.subtract(f, y)
    return np.sum(np.power(tem, 2)) / 2 / n

def gradientDesent(w, x, y, alpha):
    (n, m) = x.shape
    f = np.dot(x, w) #(n, m) . (m, 1) = (n, 1)
    w = np.subtract(w, alpha / n * np.dot(x.T, np.subtract(f, y))) #(m, n) . (n, 1); 
    #Js.append(computeCost(w, x, y))
    return w

def test(w, mean, ptp):
    path = 'test_sample.csv'
    with open(path) as f:
        testdata = pd.read_csv(f)
    testbackup = testdata.copy(deep = False)
    testdata['sex']=testdata['sex'].map(lambda x:1 if x == 'female' else 0)
    testdata['smoker']=testdata['smoker'].map(lambda x : 1 if x == 'yes' else 0)
    testdata.insert(len(testdata.columns)-1,'northeast', testdata['region'].map(lambda x : 1 if x == 'northeast' else 0))
    testdata.insert(len(testdata.columns)-1,'northwest', testdata['region'].map(lambda x : 1 if x == 'northwest' else 0))
    testdata.insert(len(testdata.columns)-1,'southeast', testdata['region'].map(lambda x : 1 if x == 'southeast' else 0))
    testdata.insert(len(testdata.columns)-1,'southwest', testdata['region'].map(lambda x : 1 if x == 'southwest' else 0))
    need = ['age','sex','bmi','children','smoker', 'northeast', 'northwest', 'southeast', 'southwest']
    for i in need:
        for j in need:
            for k in need:
                testdata.insert(len(testdata.columns)-1,i+j+k, testdata[i].mul(testdata[j]).mul(testdata[k]))
    testdata = testdata.drop(columns=['region'])
    testarray = np.array(testdata, dtype = np.float64)
    (n, m) = testarray.shape
    testx = np.divide(np.subtract(testarray[:, :m-1], mean), ptp)
    testx = np.concatenate((np.ones([n, 1], dtype = float), testx), axis = 1) 
    testresult = np.dot(testx, w).flatten()
    testbackup['charges'] = pd.Series(testresult)
    path = 'submission.csv'
    testbackup.to_csv(path, index = False)

def featureScaling(x):
    mean = np.mean(x, 0)
    ptp = np.ptp(x, axis = 0)+1e-5
    return np.divide(np.subtract(x, mean), ptp), mean, ptp

def train(data_train):
    (n, m) = data_train.shape
    w = np.random.randn(m,1)
    x, mean, ptp = featureScaling(data_train[:, :m-1])
    x = np.concatenate((np.ones([n, 1], dtype = float), x), axis = 1) 
    y = data_train[:, m-1:m]
    #Js = []
    alpha = 0.1
    iterations = 1000
    for i in range(iterations):
        w = gradientDesent(w, x, y, alpha)
    return w, mean, ptp

def val(data_val, w, mean, ptp):
    (nval, mval) = data_val.shape
    valx = np.divide(np.subtract(data_val[:, :mval-1], mean), ptp)
    valx = np.concatenate((np.ones([nval, 1], dtype = float), valx), axis = 1) 
    valy = data_val[:, mval-1:mval]
    val_lost = computeCost(w, valx, valy)
    return val_lost
    
data = getData()
val_lost_list = []
'''for i in range(10):
    (n, m) = data.shape
    data_val = data[i*n//10:(i+1)*n//10,]
    data_train = np.concatenate((data[:i*n//10,],data[(i+1)*n//10:,]), axis = 0)
    w, mean, ptp = train(data_train)
    val_lost_list.append(val(data_val, w, mean, ptp))
print(sum(val_lost_list)/len(val_lost_list))'''
#plt.plot(range(iterations), Js, 'r')
w, mean, ptp = train(data)
test(w, mean, ptp)