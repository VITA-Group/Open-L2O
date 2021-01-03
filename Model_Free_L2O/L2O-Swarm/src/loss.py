import pickle
import numpy as np 
import matplotlib.pyplot as plt
with open('./quadratic/eval_record.pickle','rb') as loss:
    data = pickle.load(loss)
print('Mat_record',len(data['Mat_record']))
#print('bias',data['inter_gradient_record'])
#print('constant',data['intra_record'])


with open('./quadratic/evaluate_record.pickle','rb') as loss1:
    data1 = pickle.load(loss1)
x = np.array(data1['x_record'])
print('x_record',x.shape)
#print('bias',data1['inter_gradient_record'])
#print('constant',data1['intra_record'])
#x = range(10000)
#ax = plt.axes(yscale='log')
#ax.plot(x,data,'b')
#plt.show('loss')