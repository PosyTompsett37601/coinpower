# -*- coding: utf-8-*-
# 只能实现对一整句话的分析，判别消极还是积极  可以用'基于语义规则'的方法，加入语义信息，然后再用lstm训练
from __future__ import absolute_import  # 把下一个新版本的特性导入到当前版本，导入3.x的特征函数
from __future__ import print_function

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

print('给语料并贴标签并合并')
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料，根据列字段对齐
# print('贴完标签、合并后的语料pn:\n',pn)
# print(pn)
neglen=len(neg)
poslen=len(pos) #计算语料数目

print('jieba分词')
# 用jieba分词
cw = lambda x: list(jieba.cut(x)) #定义分词函数, x = list(jieba.cut(x)),分词结果保存在变量x中
pn['words'] = pn[0].apply(cw)  # 对语料中的每句话都用jieba进行分词
# print('pn[words]\n',pn['words'])

comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词


d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)

print('将所有词语整合在一起……')
w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)


print('统计词的出现次数……')
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))  # dict['id']：字 该字出现频率的排名(从高到低排)
# print('dict[\'id\']:\n',dict['id'])

get_sent = lambda x: list(dict['id'][x]) # dict中'id'字段对应训练集内容，并把它从dict转化为一个list
pn['sent'] = pn['words'].apply(get_sent) #速度太慢  把评论内容拼接到一起，变成一个list
print('拼接后的pn[sent]:\n',pn['sent'])
maxlen = 50

pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2] #训练集 y存放标注
y = np.array(list(pn['mark']))[::2]
print('x:\n',x,'\ny:\n',y)
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))
print('测试集xt:\n',xt,'测试集yt',yt)
print('全集xa:\n',xa,'全集ya',ya)

print('建模中...')
model = Sequential()  #Sequential是多个网络层的线性堆叠，可以通过向Sequential模型传递一个layer的list来构造该模型：通过.add()方法一个个的将layer加入模型中
model.add(Embedding(len(dict)+1, 256))
# model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dropout(0.5))
# model.add(Dense(128, 1))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print('训练中……')
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
# model.fit(xa, ya, batch_size=16, nb_epoch=5) #训练时间为若干个小时

classes = model.predict_classes(xa)
# acc = np_utils.accuracy(classes, ya)

model.fit(xa, ya, batch_size=16, nb_epoch=2,validation_data=(xa, ya))  #迭代两次
score, acc = model.evaluate(xa, ya,batch_size=16)

print('Test score:', score)
print('Test accuracy:', acc)