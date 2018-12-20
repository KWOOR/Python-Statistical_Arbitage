# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:44:45 2018

@author: 우람
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import scale
from scipy import stats
import os
import sys
import statsmodels.api as sm

''' 기본 셋팅 '''

os.chdir('C:\\Users\\우람\\Desktop\\kaist\\4차학기\\통차')

Close=pd.read_excel('close.xlsx', index_col='Date')
Open=pd.read_excel('open.xlsx', index_col='Date')
Sma=pd.read_csv('sma14.csv', index_col='Date')
Sma.index=pd.to_datetime(Sma.index)
Wma=pd.read_excel('WMA.xlsx', index_col='Date')
Rsi=pd.read_excel('rsi.xlsx', index_col='Date')

''' 기간 나눠서.. 테스트 기간, 트레인 기간 진행!!'''


returns = Close.pct_change()
returns = returns.iloc[1:,:]

#returns_copy=returns
returns=returns.iloc[:int(len(returns)*0.8),:]
returns=returns.dropna(axis=1)


#%%

''' PCA, PEE 그래프 그려보기'''

X=returns.values
X=scale(X)

pca = PCA(n_components=100)

pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)

#%%

''' PCA , DBSCAN으로 클러스터링 (후보군 찾기) '''


N_PRIN_COMPONENTS = 40
pca = PCA(n_components=N_PRIN_COMPONENTS)
pca.fit(returns)

pca.components_.T.shape

X=np.array(pca.components_.T)
X = preprocessing.StandardScaler().fit_transform(X)

#X=np.array(returns.T) #이거로 해볼까..??

clf = DBSCAN(eps=1.8, min_samples=3)


clf.fit(X)
labels = clf.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)   #숫자 5, 총 5개의 클러스터 있음

print ("\nClusters discovered: %d" % n_clusters_)

clustered = clf.labels_   #{-1,-1,.....0....4... } 총 200개 사이즈

ticker_count = len(returns.columns) 

clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series_all = pd.Series(index=returns.columns, data=clustered.flatten())
clustered_series = clustered_series[clustered_series != -1]



CLUSTER_SIZE_LIMIT = 9999
counts = clustered_series.value_counts() #한 클러스터에 몇 개의 주식이 있는가 
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
print ("Clusters formed: %d" % len(ticker_count_reduced))
print ("Pairs to evaluate: %d" % (ticker_count_reduced*(ticker_count_reduced-1)).sum())

X_tsne = TSNE(learning_rate=1000, perplexity=25, random_state=1337).fit_transform(X)


plt.figure(1, facecolor='white')
plt.clf()
plt.axis('off')

plt.scatter(
    X_tsne[(labels!=-1), 0],
    X_tsne[(labels!=-1), 1],
    s=100,
    alpha=0.85,
    c=labels[labels!=-1],
    cmap=cm.Paired
)

plt.scatter(
    X_tsne[(clustered_series_all==-1).values, 0],
    X_tsne[(clustered_series_all==-1).values, 1],
    s=100,
    alpha=0.05
)

plt.title('T-SNE of all Stocks with DBSCAN Clusters Noted');



plt.barh(range(len(clustered_series.value_counts())),clustered_series.value_counts())
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number');


counts = clustered_series.value_counts()

# let's visualize some clusters
cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]  #집단 이름..? 집단 숫자..

# plot a handful of the smallest clusters
#for clust in cluster_vis_list[0:min(len(cluster_vis_list), 3)]:
#    tickers = list(clustered_series[clustered_series==clust].index)
#    means = np.log(Close[tickers].mean())
#    data = np.log(Close[tickers]).sub(means)
#    data.plot(title='Stock Time Series for Cluster %d' % clust)

for clust in cluster_vis_list[0:len(cluster_vis_list)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(Close[tickers].mean())
    data = np.log(Close[tickers]).sub(means)
    data.plot(title='Stock Time Series for Cluster %d' % clust)

#%%
    
''' Cointegrate 이용한 Pair 찾기 (Mean - reverting 찾기)'''

which_cluster = clustered_series.loc['DUK UN Equity']  
clustered_series[clustered_series == which_cluster]


tickers = list(clustered_series[clustered_series==which_cluster].index)
means = np.log(Close[tickers].mean())
data = np.log(Close[tickers]).sub(means)
data.plot(legend=False, title="Stock Time Series for Cluster %d" % which_cluster);


def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs


cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
        Close[tickers]
    )
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs


pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])



print ("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))
print(pairs)


stocks = np.unique(pairs)
X_df = pd.DataFrame(index=returns.T.index, data=X)
in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.loc[stocks]

X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

#plt.figure(1, facecolor='white')
#plt.clf()
#plt.axis('off')
#for pair in pairs:
#    ticker1 = pair[0]
#    loc1 = X_pairs.index.get_loc(pair[0])
#    x1, y1 = X_tsne[loc1, :]
#
#    ticker2 = pair[0]
#    loc2 = X_pairs.index.get_loc(pair[1])
#    x2, y2 = X_tsne[loc2, :]
#      
#    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');
#        
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=[in_pairs_series.values], cmap=cm.Paired)
#plt.title('T-SNE Visualization of Validated Pairs');




#%%
''' 혹시 몰라서... 페어 저장 '''
#pairs1=[('HON UN Equity', 'MMC UN Equity'), ('HON UN Equity', 'AON UN Equity'), ('BBT UN Equity', 'STI UN Equity'), ('DUK UN Equity', 'SRE UN Equity'), ('LMT UN Equity', 'NOC UN Equity')]



'찾은 페어들의 종가 그래프와 스프레드 그리기 '''

for i in range(len(pairs)):
    Close[list(pairs[i])].plot()
    (Close[pairs[i][0]]-Close[pairs[i][1]]).plot()



#%%
''' Spread 모델을 이용하기 위한 회귀분석 '''
    
''' T-Score 계산'''
    
model_close=Close[int(len(Close)*0.8):]
model_open=Open[int(len(Open)*0.8):]

model_wma_close=Wma[int(len(Wma)*0.8):]
model_wma_open=model_wma_close.shift(1)

model_sma_close=Sma[int(len(Sma)*0.8):]
model_sma_open=model_sma_close.shift(1)

model_rsi_close=Rsi[int(len(Rsi)*0.8):]
model_rsi_open=model_rsi_close.shift(1)



def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    results = sm.OLS(y, X).fit()
    return results

mod = sys.modules[__name__]


for i in range(len(pairs)):
    d_A = model_close[pairs[i][0]]-model_open[pairs[i][0]]
    d_B= model_close[pairs[i][1]]-model_open[pairs[i][1]]
    Y=d_A/model_open[pairs[i][0]]
    X=d_B/model_open[pairs[i][1]]
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
    setattr(mod, 'T_price_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] )
    setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <= 0.25* abs(X_t))*2-1).iloc[1:] )



for i in range(len(pairs)):
    d_A = model_wma_close[pairs[i][0]] - model_wma_open[pairs[i][0]]
    d_B=  model_wma_close[pairs[i][1]] - model_wma_open[pairs[i][1]]
    Y= (d_A/model_wma_open[pairs[i][0]]).dropna()
    X= (d_B/model_wma_open[pairs[i][1]]).dropna()
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
    setattr(mod, 'T_wma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )


for i in range(len(pairs)):
    d_A = model_sma_close[pairs[i][0]] - model_sma_open[pairs[i][0]]
    d_B=  model_sma_close[pairs[i][1]] - model_sma_open[pairs[i][1]]
    Y= (d_A/model_sma_open[pairs[i][0]]).dropna()
    X= (d_B/model_sma_open[pairs[i][1]]).dropna()
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
    setattr(mod, 'T_sma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )




for i in range(len(pairs)):
    d_A = model_rsi_close[pairs[i][0]] - model_rsi_open[pairs[i][0]]
    d_B=  model_rsi_close[pairs[i][1]] - model_rsi_open[pairs[i][1]]
    Y= (d_A/model_rsi_open[pairs[i][0]]).dropna()
    X= (d_B/model_rsi_open[pairs[i][1]]).dropna()
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
    setattr(mod, 'T_rsi_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )



#%%

import tensorflow as tf

keep_prob=tf.placeholder(tf.float32)

X=tf.placeholder(tf.float32, [None, 4])
Y=tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, 2)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 2])

W1=tf.get_variable("W1", shape=[4, 256], initializer=tf.contrib.layers.xavier_initializer())
b1= tf.Variable(tf.random_normal([256]))
L1= tf.nn.relu(tf.matmul(X,W1)+b1)
L1= tf.nn.dropout(L1, keep_prob=keep_prob)

W2=tf.get_variable("W2", shape=[256, 128],  initializer=tf.contrib.layers.xavier_initializer())
b2= tf.Variable(tf.random_normal([128]))
L2= tf.nn.relu(tf.matmul(L1,W2)+b2)
L2= tf.nn.dropout(L2, keep_prob=keep_prob)

W3=tf.get_variable("W3", shape=[128, 2], initializer=tf.contrib.layers.xavier_initializer())
b3= tf.Variable(tf.random_normal([2]))
hypothesis= tf.matmul(L2,W3)+b3 

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 


predicton = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(predicton, tf.arg_max(Y_one_hot,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    a=pd.concat([T_wma_0, T_sma_0, T_rsi_0, T_price_0], axis=1)
    train_x= np.array(a[:int(len(a)*0.8)])
    test_x= np.array(a[int(len(a)*0.8):])
    train_y=np.array(label_0[:int(len(label_0)*0.8)])
    test_y=np.array(label_0[int(len(label_0)*0.8):])
   
    for epoch in range(15):
        avg_cost=0
        for i in range(100):
            c , _=sess.run([cost, optimizer], feed_dict={X:train_x, Y:train_y, keep_prob:1}) #여긴 0.7
            avg_cost += c/100
        print("epoch:", '%04d' %(epoch +1), "cost=", "{:.9f}".format(avg_cost))

    #이 밑으론 실전이니까 keep_prob:1 이 되어야함!!    
    print("Accuracy", accuracy.eval(session=sess, feed_dict={X: test_x, Y:test_y, keep_prob:1})) #여긴 1





tf.reset_default_graph()





















