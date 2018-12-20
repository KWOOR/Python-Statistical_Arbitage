# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:19:54 2018

@author: 우람
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:52:55 2018

@author: 우람
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:44:45 2018

@author: 우람
"""
#%%
#MFI 만드는 함수
'''
def MoneyFlowIndex_Ndays(close_price,open_price,volume_,n):
    
    save_index = pd.Series(close_p.index)
    close_price = close_p.reset_index(drop=True)
    open_price = open_p.reset_index(drop=True)
    volume_ = volume.reset_index(drop=True)
     
    typical_price = (open_price + close_price)/2

    for columns in typical_price.columns :
        
        for j in range(len(typical_price)-1):
            
            if typical_price.loc[j+1,columns] > typical_price.loc[j,columns]:
                
               typical_price.loc[j+1,columns] = typical_price.loc[j+1,columns]*1
               
            else:
               
               typical_price.loc[j+1,columns] = typical_price.loc[j+1,columns]*(-1)
              
                
    RawMoneyFlow = typical_price*volume_     
    
    pos = (RawMoneyFlow>0)*1
    pos = pos.astype(int)
    
    neg = (RawMoneyFlow<0)*1
    neg = neg.astype(int)
    
    P_MoneyFlow = RawMoneyFlow*pos 
    N_MoneyFlow = RawMoneyFlow*neg  
    
    posflow_sum = pd.DataFrame(np.zeros((9,14)))
    negflow_sum = pd.DataFrame(np.zeros((9,14)))
    
    posflow_sum_ = pd.DataFrame()
    negflow_sum_ = pd.DataFrame()
    
    for j in range(14,len(P_MoneyFlow)):
   
       P_MoneyFlow_ =  P_MoneyFlow.loc[j-14:j,:].sum(axis=0)
       N_MoneyFlow_ =  N_MoneyFlow.loc[j-14:j,:].sum(axis=0)
       posflow_sum_ = pd.concat([posflow_sum_,P_MoneyFlow_],axis=1)
       negflow_sum_ = pd.concat([negflow_sum_,N_MoneyFlow_],axis=1)
    
    posflow_sum = posflow_sum.set_index(posflow_sum_.index)
    negflow_sum = negflow_sum.set_index(negflow_sum_.index)
    PositiveMoneyFlow = pd.concat([posflow_sum,posflow_sum_],axis=1,join='inner').T.set_index(save_index)
    NegativeMoneyFlow = pd.concat([negflow_sum,np.abs(negflow_sum_)],axis=1,join='inner').T.set_index(save_index) 

        
    MoneyFlowRatio = PositiveMoneyFlow/NegativeMoneyFlow
    
    MoneyFlowIndex = 100-100/(1+MoneyFlowRatio)
    
    return MoneyFlowIndex
'''

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import sys
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve, auc,accuracy_score
from sklearn.metrics import classification_report



''' 기본 셋팅 '''

os.chdir('C:\\Users\\우람\\Desktop\\kaist\\4차학기\\통차')

Close=pd.read_excel('close.xlsx', index_col='Date')
Open=pd.read_excel('open.xlsx', index_col='Date')
Sma=pd.read_csv('sma14.csv', index_col='Date')
Sma.index=pd.to_datetime(Sma.index)
Wma=pd.read_excel('WMA.xlsx', index_col='Date')
Mfi=pd.read_csv('MFI14.csv', index_col='Date')
Mfi.index=pd.to_datetime(Mfi.index)
Rsi=pd.read_excel('rsi.xlsx', index_col='Date')


def MinMaxScale(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)



''' 기간 나눠서.. 테스트 기간, 트레인 기간 진행!!'''


returns = Close.pct_change()
returns = returns.iloc[1:,:]

#returns_copy=returns

'''Formation Period를 전체의 80%로 둔다'''

returns=returns.iloc[:int(len(returns)*0.8),:]  #Formation Period 동안의 수익률로 페어 찾고 분석 
returns=returns.dropna(axis=1)


#%%

''' PCA, PEE 그래프 그려보기'''

X=returns.T.values
X=scale(X)

pca = PCA(n_components=100)

pca.fit(X)

var= pca.explained_variance_ratio_

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.title('PCA explained variance ratio')
plt.ylabel('Explained Percentage(%)')
plt.xlabel('Number of components')
plt.show()


#%%

''' PCA , DBSCAN으로 클러스터링 (후보군 찾기) '''


N_PRIN_COMPONENTS = 40
pca = PCA(n_components=N_PRIN_COMPONENTS)
pca.fit(returns)
#pca2=PCA(n_components=1)
#pca3=PCA(n_components=40)
#
#''' 마켓 포트폴리오 비중 보고싶을 때'''
#w = pca2.fit_transform(returns.T)
#m = pca2.mean_
#p1 = pca2.components_[0]
#df_w = pd.DataFrame(w)
#df_w.index = returns.columns
#df_w.columns = ["주성분 비중"]
#df_w
#
#''' 뭔진 모르겠지만 우리가 원하는 팩터의 비중 보고 싶을 때 '''
#w = pca2.fit_transform(returns)
#m = pca2.mean_
#p1 = pca2.components_[0]
#df_w = pd.DataFrame(w)
#df_w.index = returns.index
#df_w.columns = ["주성분 비중"]
#df_w
#
#w=pca3.fit(returns)
#x=pca3.components_

pca.components_.T.shape
#len(pca.components_[0])
X=np.array(pca.components_.T)
X = preprocessing.StandardScaler().fit_transform(X)

#X=pd.DataFrame(X.T)
#X.columns=returns.columns
#X['JPM UN Equity'].plot()
#X['WFC UN Equity'].plot()
#X['GS UN Equity'].plot()
#X['BBT UN Equity'].plot()
#X['STI UN Equity'].plot()
#Close['BBT UN Equity'].plot()
#Close['STI UN Equity'].plot()
#X['LMT UN Equity'].plot()
#X['NOC UN Equity'].plot()
#Close['LMT UN Equity'].plot()
#Close['NOC UN Equity'].plot()
#X.plot()
clf = DBSCAN(eps=1.5, min_samples=3)


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


#
#A=np.array([[1,0,0], [0,1,0], [0,0,1]])
#pca = PCA()
#
#pca.fit(A)
#pca.explained_variance_
#pca.components_[0]
#a=pca.fit(A)
#
#cov=returns.cov()
#eigenv=np.array([pca.components_[0]])
#np.dot(cov ,eigenv.T)
#len(np.dot(cov ,eigenv.T))
#pca.singular_values_[0]* eigenv.T
#
#
#a=np.dot(cov ,eigenv.T)
#b=eig(cov)[0][0] * eigenv.T
#
#eig = np.linalg.eig
#eigcov=eig(cov)[1]
#eig(A.cov())[0][0]
#from sklearn.preprocessing import StandardScaler
#a=StandardScaler().fit_transform(returns)
#pca.fit(a)
#a=pca.components_.T*np.sqrt(np.array([pca.explained_variance_]))
#a=StandardScaler().fit_transform(a)

#%%

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

#%%

plt.barh(range(len(clustered_series.value_counts())),clustered_series.value_counts())
plt.title('Cluster Member Counts')
plt.xlabel('Stocks in Cluster')
plt.ylabel('Cluster Number');


counts = clustered_series.value_counts()

# let's visualize some clusters
cluster_vis_list = list(counts[(counts<20) & (counts>1)].index)[::-1]  #집단 이름..? 집단 숫자..


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

#X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)
#
##plt.figure(1, facecolor='white')
##plt.clf()
##plt.axis('off')
##for pair in pairs:
##    ticker1 = pair[0]
##    loc1 = X_pairs.index.get_loc(pair[0])
##    x1, y1 = X_tsne[loc1, :]
##
##    ticker2 = pair[0]
##    loc2 = X_pairs.index.get_loc(pair[1])
##    x2, y2 = X_tsne[loc2, :]
##      
##    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray');
##        
##plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=[in_pairs_series.values], cmap=cm.Paired)
##plt.title('T-SNE Visualization of Validated Pairs');
#



#%%
''' 혹시 몰라서... 페어 저장 '''
pairs=[('HON UN Equity', 'MMC UN Equity'), ('HON UN Equity', 'AON UN Equity'), ('BBT UN Equity', 'STI UN Equity'), ('DUK UN Equity', 'SRE UN Equity'), ('LMT UN Equity', 'NOC UN Equity')]



'찾은 페어들의 종가 그래프와 스프레드 그리기 '''

for i in range(len(pairs)):
    Close[list(pairs[i])].plot()
    plt.axvline(x='2017-09-08')
    ax = (Close[pairs[i][0]]-Close[pairs[i][1]]).plot(title='Stock price of pairs and spread')
    ax.legend(['{}'.format(pairs[i][0]),'{}'.format(pairs[i][1]),'Spread'])


#%%
''' Spread 모델을 이용하기 위한 회귀분석 '''
    
''' T-Score 계산'''
''' 스프레드가 줄어들지 늘어날지에 대한 방향성만 보기 위해 절대값 사용'''
    
model_close=Close
model_open=Open

model_wma_close=Wma
model_wma_open=model_wma_close.shift(1)

model_sma_close=Sma
model_sma_open=model_sma_close.shift(1)

model_rsi_close=Rsi
model_rsi_open=model_rsi_close.shift(1)

model_mfi_close=Mfi
model_mfi_open=model_mfi_close.shift(1)


def reg_m(y, x):
    ones = np.ones(len(x))
    X = sm.add_constant(np.column_stack((x, ones)))
    results = sm.OLS(y, X).fit()
    return results

mod = sys.modules[__name__]

beta_=[]
empty=[]
for i in range(len(pairs)):
    d_A = model_close[pairs[i][0]]-model_open[pairs[i][0]]
    d_B= model_close[pairs[i][1]]-model_open[pairs[i][1]]
    Y=d_A/model_open[pairs[i][0]]
    X=d_B/model_open[pairs[i][1]]
#    Y=Y.fillna(0)
#    X=X.fillna(0)
#    Y=Y.dropna()
#    X=X.dropna()
    result=reg_m(Y,X)
    beta_.append(result.params[0])
    print(result.summary())  #Durbin - Watson은 에러텀의 독립성.. 시계열 데이터 분석할 때 쓰지.. 
    #첫 페어에서는 2에 가까웠기에 AR(1) 모형에서의 beta값이 조금 그럼..
    # beta의 유의성이 뛰어남!! 
    X_t=result.resid
    spread=Close[pairs[i][0]]-Close[pairs[i][1]]
#    resid=(X_t-X_t.shift(1)).fillna(0)
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    print(result.summary())
    #AR(1) 모형으로 했는데 Durbin-Watson은 2에 가까움. 추정 잘 됨. beta값도 marginal하게 유의함!(마지막 페어 빼고)
    # 여기서의 x1(=beta)는 theta를 의미함. 즉 평균회귀속도 const=theta*mean 인데.. 거의 0이지.. long-term mean도 거의 0으로 봐도..
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
#    
#    buy_open= (pd.DataFrame(X_t >0) & pd.DataFrame(X_t.shift(-1) < X_t))*1
#    sell_open= (pd.DataFrame(X_t <0) & pd.DataFrame(X_t.shift(-1) > X_t))*-1
    buy_open= (pd.DataFrame(X_t >X_t.mean()) & pd.DataFrame(X_t.shift(-1) < 0.8*X_t))*1
    sell_open= (pd.DataFrame(X_t < X_t.mean()) & pd.DataFrame(X_t.shift(-1) > 0.8*X_t))*-1
#    buy_open = (pd.DataFrame(X_t >0) * pd.DataFrame(X_t.shift(1) > 1.2*X_t))*1
#    sell_open = (pd.DataFrame(X_t <0) * pd.DataFrame(X_t.shift(1) < 1.2*X_t))*-1
#    buy_open= pd.DataFrame(X_t > 1.2*X_t.std())*1 #* pd.DataFrame(X_t.shift(-1) <1* X_t)*1
#    print(buy_open.max())
#    sell_open= pd.DataFrame(X_t < -1.2*X_t.std())*-1 #* pd.DataFrame(X_t.shift(-1) > 1*X_t)*-1
    for j in range(100):
        buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * (pd.DataFrame(X_t > 0))*1
        sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * pd.DataFrame(X_t <0))*-1  
#    for j in range(100):        
#        buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * pd.DataFrame(X_t > X_t.mean()-0.05*X_t.std())*1
#        sell_open-=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * pd.DataFrame(X_t < X_t.mean()+ 0.05*X_t.std())*1  
#    sell_open-=(sell_open==0)* (sell_open.shift(1)==-1) * pd.DataFrame(X_t < X_t.mean())
#    print(sell_open.min())
    
#    setattr(mod, 'T_price_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] )
    setattr(mod, 'T_price_{}'.format(i), pd.DataFrame( (X_t - mu) /sigma ).iloc[1:] )
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) < 0.5* abs(X_t))*2-1).iloc[1:] )
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <  abs(X_t))*2-1).iloc[1:] )
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame( buy_open+sell_open+buy_close+sell_close).iloc[1:] ) 
    setattr(mod, 'label_{}'.format(i), pd.DataFrame( buy_open+sell_open,dtype='i').iloc[1:] ) 
    empty.append(X_t)
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame( (spread.shift(-1) < spread)    *2-1).iloc[1:] )


for i in range(len(pairs)):
    d_A = model_wma_close[pairs[i][0]] - model_wma_open[pairs[i][0]]
    d_B=  model_wma_close[pairs[i][1]] - model_wma_open[pairs[i][1]]
    Y= (d_A/model_wma_open[pairs[i][0]]).dropna()
    X= (d_B/model_wma_open[pairs[i][1]]).dropna()
#    Y=Y.fillna(0)
#    X=X.fillna(0)
#    Y=Y.dropna()
#    X=X.dropna()
    result=reg_m(Y,X)
#    print(result.summary())
    X_t=result.resid
    
    resid=result.resid
#    resid=(X_t-X_t.shift(1)).fillna(0)
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
#    print(result.summary())

    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
#    setattr(mod, 'T_wma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
    setattr(mod, 'T_wma_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ) )



for i in range(len(pairs)):
    d_A = model_sma_close[pairs[i][0]] - model_sma_open[pairs[i][0]]
    d_B=  model_sma_close[pairs[i][1]] - model_sma_open[pairs[i][1]]
    Y= (d_A/model_sma_open[pairs[i][0]]).dropna()
    X= (d_B/model_sma_open[pairs[i][1]]).dropna()
#    Y=Y.fillna(0)
#    X=X.fillna(0)
#    Y=Y.dropna()
#    X=X.dropna()
    result=reg_m(Y,X)
#    print(result.summary())
    X_t=result.resid
    
    resid=result.resid
#    resid=(X_t-X_t.shift(1)).fillna(0)
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
#    print(result.summary())
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
#    setattr(mod, 'T_sma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
    setattr(mod, 'T_sma_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ) )




for i in range(len(pairs)):
    d_A = model_rsi_close[pairs[i][0]] - model_rsi_open[pairs[i][0]]
    d_B=  model_rsi_close[pairs[i][1]] - model_rsi_open[pairs[i][1]]
    Y= (d_A/model_rsi_open[pairs[i][0]]).dropna()
    X= (d_B/model_rsi_open[pairs[i][1]]).dropna()
#    Y=Y.fillna(0)
#    X=X.fillna(0)
#    Y=Y.dropna()
#    X=X.dropna()
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
#    resid=(X_t-X_t.shift(1)).fillna(0)
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
#    setattr(mod, 'T_rsi_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
    setattr(mod, 'T_rsi_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ) )



for i in range(len(pairs)):
    d_A = model_mfi_close[pairs[i][0]] - model_mfi_open[pairs[i][0]]
    d_B=  model_mfi_close[pairs[i][1]] - model_mfi_open[pairs[i][1]]
    Y= (d_A/model_mfi_open[pairs[i][0]]).dropna()
    X= (d_B/model_mfi_open[pairs[i][1]]).dropna()
#    Y=Y.fillna(0)
#    X=X.fillna(0)
#    Y=Y.dropna()
#    X=X.dropna()
    result=reg_m(Y,X)
    X_t=result.resid
    
    resid=result.resid
#    resid=(X_t-X_t.shift(1)).fillna(0)
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
    
#    setattr(mod, 'T_mfi_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
    setattr(mod, 'T_mfi_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ) )

    
    #%%
    '''위의 내용을 함수로 만든 것 '''
     
def Caculate_Tscore(case):
    
    if case == 1:                        
    
       data1_close = model_close
       data2_open = model_open 
    
    elif case == 2:                       
         
         data1_close = model_wma_close
         data2_open = model_wma_open 
        
    elif case == 3:                       
         data1_close = model_sma_close
         data2_open = model_sma_open 
         
    elif case == 4:                       
         data1_close = model_rsi_close
         data2_open = model_rsi_open    
         
    elif case == 5:                       
         data1_close = model_mfi_close
         data2_open = model_mfi_open    
    
    for i in range(len(pairs)):
        
        d_A = data1_close[pairs[i][0]]-data2_open[pairs[i][0]]
        d_B= data1_close[pairs[i][1]]-data2_open[pairs[i][1]]   
        
        if case == 1: 
                
           Y= (d_A/data2_open[pairs[i][0]])
           X= (d_B/data2_open[pairs[i][1]])
            
        else:
            
            Y= (d_A/data2_open[pairs[i][0]]).dropna()
            X= (d_B/data2_open[pairs[i][1]]).dropna()
             
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
        
        if case == 1:
            
           setattr(mod, 'T_Price_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] )
           setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <= 0.25* abs(X_t))*2-1).iloc[1:] ) 
           
        if case == 2:
            
           setattr(mod, 'T_wma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
           
        if case == 3:   
          
           setattr(mod, 'T_sma_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
           
        if case == 4:
            
           setattr(mod, 'T_rsi_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
           
        if case == 5:
            
           setattr(mod, 'T_mfi_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
          
    return 

for i in range(1,6):
  
    Caculate_Tscore(i)
    
#%%
    ''' Data Concat 및 학습, 테스트 데이터로 분류하기 '''
    
Pair1 = pd.concat([T_price_0.shift(1).fillna(0),T_price_0,T_mfi_0,T_rsi_0,T_sma_0,T_wma_0,label_0],axis=1,join='inner').values

Pair2 = pd.concat([T_price_1.shift(1).fillna(0),T_price_1,T_mfi_1,T_rsi_1,T_sma_1,T_wma_1,label_1],axis=1,join='inner').values

Pair3 = pd.concat([T_price_2.shift(1).fillna(0),T_price_2,T_mfi_2,T_rsi_2,T_sma_2,T_wma_2,label_2],axis=1,join='inner').values

Pair4 = pd.concat([T_price_3.shift(1).fillna(0),T_price_3,T_mfi_3,T_rsi_3,T_sma_3,T_wma_3,label_3],axis=1,join='inner').values

Pair5 = pd.concat([T_price_4.shift(1).fillna(0),T_price_4,T_mfi_4,T_rsi_4,T_sma_4,T_wma_4,label_4],axis=1,join='inner').values
    

Pair1X_train, Pair1X_test, Pair1Y_train, Pair1Y_test = train_test_split(Pair1[:,0:5], Pair1[:,-1],
                                                                        test_size=0.2)

Pair2X_train, Pair2X_test, Pair2Y_train, Pair2Y_test = train_test_split(Pair2[:,0:5], Pair2[:,-1],
                                                                        test_size=0.2)

Pair3X_train, Pair3X_test, Pair3Y_train, Pair3Y_test = train_test_split(Pair3[:,0:5], Pair3[:,-1],
                                                                        test_size=0.2)

Pair4X_train, Pair4X_test, Pair4Y_train, y_pred4 = train_test_split(Pair4[:,0:5], Pair4[:,-1],
                                                                        test_size=0.2)

Pair5X_train, Pair5X_test, Pair5Y_train, Pair5Y_test = train_test_split(Pair5[:,0:5], Pair5[:,-1],
                                                                        test_size=0.2) #random_state=100 으로 두면, 섞어서 배출해준다. 그러나 섞이면 안 되므로 안 씀


Pair1X_train, Pair1X_test, Pair1Y_train, Pair1Y_test = Pair1[:int(len(Pair1)*0.8),:-1] ,  Pair1[int(len(Pair1)*0.8):,:-1], Pair1[:int(len(Pair1)*0.8),-1] ,  Pair1[int(len(Pair1)*0.8):,-1]
Pair2X_train, Pair2X_test, Pair2Y_train, Pair2Y_test = Pair2[:int(len(Pair2)*0.8),:-1] ,  Pair2[int(len(Pair2)*0.8):,:-1], Pair2[:int(len(Pair2)*0.8),-1] ,  Pair2[int(len(Pair2)*0.8):,-1]
Pair3X_train, Pair3X_test, Pair3Y_train, Pair3Y_test = Pair3[:int(len(Pair3)*0.8),:-1] ,  Pair3[int(len(Pair3)*0.8):,:-1], Pair3[:int(len(Pair3)*0.8),-1] ,  Pair3[int(len(Pair3)*0.8):,-1]
Pair4X_train, Pair4X_test, Pair4Y_train, Pair4Y_test = Pair4[:int(len(Pair4)*0.8),:-1] ,  Pair4[int(len(Pair4)*0.8):,:-1], Pair4[:int(len(Pair4)*0.8),-1] ,  Pair4[int(len(Pair4)*0.8):,-1]
Pair5X_train, Pair5X_test, Pair5Y_train, Pair5Y_test = Pair5[:int(len(Pair5)*0.8),:-1] ,  Pair5[int(len(Pair5)*0.8):,:-1], Pair5[:int(len(Pair5)*0.8),-1] ,  Pair5[int(len(Pair5)*0.8):,-1]


#%% NN

import tensorflow as tf

tf.reset_default_graph()

keep_prob=tf.placeholder(tf.float32)

X=tf.placeholder(tf.float32, [None, 5])
Y=tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, 3)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 3])

W1=tf.get_variable("W1", shape=[5, 3], initializer=tf.contrib.layers.xavier_initializer())
b1= tf.Variable(tf.random_normal([3]))
L1= tf.nn.relu(tf.matmul(X,W1)+b1)
L1= tf.nn.dropout(L1, keep_prob=keep_prob)
#
#W2=tf.get_variable("W2", shape=[20, 2],  initializer=tf.contrib.layers.xavier_initializer())
#b2= tf.Variable(tf.random_normal([2]))
#L2= tf.nn.relu(tf.matmul(L1,W2)+b2)
#L2= tf.nn.dropout(L2, keep_prob=keep_prob)
#
#W3=tf.get_variable("W3", shape=[128, 2], initializer=tf.contrib.layers.xavier_initializer())
#b3= tf.Variable(tf.random_normal([2]))
#hypothesis= tf.matmul(L2,W3)+b3 

hypothesis= tf.matmul(X,W1)+b1


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost) 


prediction = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(prediction, tf.arg_max(Y_one_hot,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))


a=pd.concat([T_wma_0, T_sma_0, T_rsi_0, T_price_0, T_mfi_0, label_0], axis=1)
a=a.dropna()
train_x= np.array(a[:int(len(a)*0.8)].iloc[:,:-1])
test_x= np.array(a[int(len(a)*0.8):].iloc[:,:-1])
train_y=np.array(a[:int(len(a)*0.8)].iloc[:,-1]).reshape([len(train_x),1])
test_y=np.array(a[int(len(a)*0.8):].iloc[:,-1]).reshape([len(test_x),1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    
    batch_size=100
    for epoch in range(10000):
        avg_cost=0
        total_batch = int(len(train_x)/batch_size)
        start = 0
        end = batch_size
        for k in range(total_batch):
            if k != total_batch-1:
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
            else:
                batch_x = train_x[start:]
                batch_y = train_y[start:] 
            start += batch_size
            end += batch_size
#            loss_val, _ = sess.run([loss, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
#            avg_loss += loss_val / total_batch
#        for i in range(50):
            c , _=sess.run([cost, optimizer], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7}) #여긴 0.7
            avg_cost += c/total_batch
        if epoch%100==0:
            print("epoch:", '%04d' %(epoch +1), "cost=", "{:.9f}".format(avg_cost))

    #이 밑으론 실전이니까 keep_prob:1 이 되어야함!!    
    print("Accuracy", accuracy.eval(session=sess, feed_dict={X: test_x, Y:test_y, keep_prob:1})) #여긴 1
    preds=prediction.eval(session=sess, feed_dict={X:test_x, keep_prob:1})

    ''' -1은 0, 1은 1, 0은 2로 반환..'''

#%%
    
''' Soft Vector Machine 사용'''
'''시그널 하나'''
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#clf = svm.SVC(gamma=10, decision_function_shape='ovr', kernel='poly')
clf = svm.LinearSVC(C=10)
#clf = svm.LinearSVC(penalty='l1', C=10, dual=False, max_iter=5000)
#clf = svm.LinearSVC(penalty='l1', C=20, dual=False, max_iter=10000, multi_class='crammer_singer')
#clf = svm.LinearSVC( C=20, max_iter=10000, multi_class='ovr', tol=0.0001)


X=Pair1X_train
Y=Pair1Y_train

buff=pd.concat([pd.DataFrame(Pair1X_train), pd.DataFrame(Pair1X_test)])
#X = preprocessing.StandardScaler().fit_transform(X)



# Scaling하고 테스트셋과 트레이닝셋 쪼개기 

X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

clf.fit(X_train , Y)  

#X=Pair1X_test
y_pred1=clf.predict(list(X_test))

print("Accuracy:",metrics.accuracy_score(Pair1Y_test, y_pred1))   
print(y_pred1)

print(((y_pred1==1)*1).sum())

conf_matrix=confusion_matrix(Pair1Y_test, y_pred1)    
print(conf_matrix)  # -1, 0, 1 순서로 출력한다 
print(classification_report(Pair1Y_test, y_pred1))
#print("Recall:", conf_matrix[1][1]/ (conf_matrix[1][1]+ conf_matrix[1][0])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) #손실 기회를 성공적으로 피함
#print("Precision:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) # 전체 거래 횟수 중에 이익을 본 거래 횟수

#print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수



Y=Pair2Y_train

buff=pd.concat([pd.DataFrame(Pair2X_train), pd.DataFrame(Pair2X_test)])
#X = preprocessing.StandardScaler().fit_transform(X)


X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

clf.fit(X_train , Y)  

#X=Pair1X_test
y_pred2=clf.predict(list(X_test))

print("Accuracy:",metrics.accuracy_score(Pair2Y_test, y_pred2))   
print(y_pred2)

print(((y_pred2==1)*1).sum())
    
conf_matrix=confusion_matrix(Pair2Y_test, y_pred2)    
print(conf_matrix)
print(classification_report(Pair2Y_test, y_pred2))

#print("Recall:", conf_matrix[1][1]/ (conf_matrix[1][1]+ conf_matrix[1][0])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) #손실 기회를 성공적으로 피함
#print("Precision:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
#    
#75.767 same
#print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수


Y=Pair3Y_train

buff=pd.concat([pd.DataFrame(Pair3X_train), pd.DataFrame(Pair3X_test)])
#X = preprocessing.StandardScaler().fit_transform(X)


X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

clf.fit(X_train , Y)  

#X=Pair1X_test
y_pred3=clf.predict(list(X_test))

print("Accuracy:",metrics.accuracy_score(Pair3Y_test, y_pred3))   
print(y_pred3)

print(((y_pred3==1)*1).sum())

conf_matrix=confusion_matrix(Pair3Y_test, y_pred3)    
print(conf_matrix)
print(classification_report(Pair3Y_test, y_pred3))

#print("Recall:", conf_matrix[1][1]/ (conf_matrix[1][1]+ conf_matrix[1][0])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) #손실 기회를 성공적으로 피함
#print("Precision:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
#    
## 71.33  71.67
#print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수


Y=Pair4Y_train

buff=pd.concat([pd.DataFrame(Pair4X_train), pd.DataFrame(Pair4X_test)])
#X = preprocessing.StandardScaler().fit_transform(X)


X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

clf.fit(X_train , Y)  

#X=Pair1X_test
y_pred4=clf.predict(list(X_test))

print("Accuracy:",metrics.accuracy_score(Pair4Y_test, y_pred4))   
print(y_pred4)

print(((y_pred4==1)*1).sum())

conf_matrix=confusion_matrix(Pair4Y_test, y_pred4)    
print(conf_matrix)
print(classification_report(Pair4Y_test, y_pred4))

#print("Recall:", conf_matrix[1][1]/ (conf_matrix[1][1]+ conf_matrix[1][0])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) #손실 기회를 성공적으로 피함
#print("Precision:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
##    
# 73.72 same
#print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수


Y=Pair5Y_train

buff=pd.concat([pd.DataFrame(Pair5X_train), pd.DataFrame(Pair5X_test)])
#X = preprocessing.StandardScaler().fit_transform(X)


X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

clf.fit(X_train , Y)  

#X=Pair1X_test
y_pred5=clf.predict(list(X_test))

print("Accuracy:",metrics.accuracy_score(Pair5Y_test, y_pred5))   
print(y_pred5)

print(((y_pred5==1)*1).sum())

conf_matrix=confusion_matrix(Pair5Y_test, y_pred5)    
print(conf_matrix)
print(classification_report(Pair5Y_test, y_pred5))

#print("Recall:", conf_matrix[1][1]/ (conf_matrix[1][1]+ conf_matrix[1][0])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])) #손실 기회를 성공적으로 피함
#print("Precision:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
##    
#print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수

# 74.40  75.08

#%% 
'''Trading Signal 그리기 '''
plt.figure(figsize = (12,8))
plt.plot(empty[0][int(len(empty[0])*0.8)+3:].index,empty[0][int(len(empty[0])*0.8)+3:],'g',linewidth=0.8,label = 'Rolling_Z_score')
plt.plot(empty[0][int(len(empty[0])*0.8)+3:].index,y_pred1*empty[0][int(len(empty[0])*0.8)+3:].std()*1,'k.',label = 'Position')
plt.plot(empty[0][int(len(empty[0])*0.8)+3:].index,np.ones(len(y_pred1))*(-empty[0][int(len(empty[0])*0.8)+3:].std()*1),'r--',label = 'Short position') 
plt.plot(empty[0][int(len(empty[0])*0.8)+3:].index,np.ones(len(y_pred1))*(empty[0][int(len(empty[0])*0.8)+3:].std()*1),'b--',label = 'Long position') 
#plt.plot(pd.DataFrame(Pair1Y_test).index,np.ones(len(Pair1Y_test))*2,'k-',linewidth=2,label = 'Short position Tresholds')
#plt.plot(pd.DataFrame(Pair1Y_test).index,np.ones(len(Pair1Y_test))*(-2),'k-',linewidth=2,label = 'Long position Tresholds')  
#plt.ylim([-3,3])
plt.legend(loc='upper left'   ) 
plt.xlabel('Date')
plt.ylabel('Positions based on Z-score')
plt.title('Overview of Trading Positions')
plt.show()


plt.figure(figsize = (12,8))
plt.plot(empty[0][14:int(len(empty[0])*0.8)].index,empty[0][14:int(len(empty[0])*0.8)],'g',linewidth=0.8,label = 'Rolling_Z_score')
plt.plot(empty[0][14:int(len(empty[0])*0.8)].index,Pair1Y_train[:-2]*empty[0][14:int(len(empty[0])*0.8)].std()*2,'k.',label = 'Position')
plt.plot(empty[0][14:int(len(empty[0])*0.8)].index,np.ones(len(Pair1Y_train[:-2]))*(-empty[0][14:int(len(empty[0])*0.8)].std()*2),'r--',label = 'Short position') 
plt.plot(empty[0][14:int(len(empty[0])*0.8)].index,np.ones(len(Pair1Y_train[:-2]))*(empty[0][14:int(len(empty[0])*0.8)].std()*2),'b--',label = 'Long position') 
#plt.plot(pd.DataFrame(Pair1Y_test).index,np.ones(len(Pair1Y_test))*2,'k-',linewidth=2,label = 'Short position Tresholds')
#plt.plot(pd.DataFrame(Pair1Y_test).index,np.ones(len(Pair1Y_test))*(-2),'k-',linewidth=2,label = 'Long position Tresholds')  
#plt.ylim([-3,3])
plt.legend(loc='upper left'   ) 
plt.xlabel('Date')
plt.ylabel('Positions based on Z-score')
plt.title('Overview of Trading Positions')
plt.show()

#%%

''' 시그널 하나일때 결과 요약 '''
answer=[]
total=[]

payoff=-1*Pair1Y_train*(Close[pairs[0][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])+Pair1Y_train*(Close[pairs[0][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])
((payoff)+1).cumprod()
payoff=-1*Pair1Y_test*(Close[pairs[0][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+Pair1Y_test*(Close[pairs[0][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
answer.append(((payoff)+1).cumprod())
print("학습기간 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
payoff=-1*y_pred1*(Close[pairs[0][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+y_pred1*(Close[pairs[0][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
total.append(((payoff)))
print("누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
print("거래비용은:", (pd.DataFrame(y_pred1).replace(-1,1).sum() - (pd.DataFrame(y_pred1).replace(0,np.nan) == pd.DataFrame(y_pred1).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
print("="*60, "\n")

payoff=-1*Pair2Y_train*(Close[pairs[1][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])+Pair2Y_train*(Close[pairs[1][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])
((payoff)+1).cumprod()
payoff=-1*Pair2Y_test*(Close[pairs[1][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+Pair2Y_test*(Close[pairs[1][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
answer.append(((payoff)+1).cumprod())
print("학습기간 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
payoff=-1*y_pred2*(Close[pairs[1][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+y_pred2*(Close[pairs[1][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
total.append(((payoff)))
print("누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
print("거래비용은:", (pd.DataFrame(y_pred2).replace(-1,1).sum() - (pd.DataFrame(y_pred2).replace(0,np.nan) == pd.DataFrame(y_pred2).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
print("="*60, "\n")

payoff=-1*Pair3Y_train*(Close[pairs[2][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])+Pair3Y_train*(Close[pairs[2][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])
((payoff)+1).cumprod()
payoff=-1*Pair3Y_test*(Close[pairs[2][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+Pair3Y_test*(Close[pairs[2][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
answer.append(((payoff)+1).cumprod())
print("학습기간 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
payoff=-1*y_pred3*(Close[pairs[2][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])+y_pred3*(Close[pairs[2][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff/2)+1).cumprod()
total.append(((payoff)))
print("누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
print("거래비용은:", (pd.DataFrame(y_pred3).replace(-1,1).sum() - (pd.DataFrame(y_pred3).replace(0,np.nan) == pd.DataFrame(y_pred3).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
print("="*60, "\n")


payoff=-1*Pair4Y_train*(Close[pairs[3][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])+Pair4Y_train*(Close[pairs[3][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])
((payoff)+1).cumprod()
payoff=-1*Pair4Y_test*(Close[pairs[3][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+Pair4Y_test*(Close[pairs[3][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
answer.append(((payoff)+1).cumprod())
print("학습기간 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
payoff=-1*y_pred4*(Close[pairs[3][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+y_pred4*(Close[pairs[3][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
total.append(((payoff)))
print("누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
print("거래비용은:", (pd.DataFrame(y_pred4).replace(-1,1).sum() - (pd.DataFrame(y_pred4).replace(0,np.nan) == pd.DataFrame(y_pred4).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
print("="*60, "\n")

payoff=-1*Pair5Y_train*(Close[pairs[4][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])+Pair5Y_train*(Close[pairs[4][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(buff)*0.8)])
((payoff)+1).cumprod()
payoff=-1*Pair5Y_test*(Close[pairs[4][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+Pair5Y_test*(Close[pairs[4][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
answer.append(((payoff)+1).cumprod())
print("학습기간 누적수익률은:", ((payoff+1).cumprod().iloc[-2]-1)*100,"%")
payoff=-1*y_pred5*(Close[pairs[4][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])+ y_pred5*(Close[pairs[4][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(buff)*0.8):])
((payoff)+1).cumprod()
total.append(((payoff)))
print("누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
print("거래비용은:", (pd.DataFrame(y_pred5).replace(-1,1).sum() - (pd.DataFrame(y_pred5).replace(0,np.nan) == pd.DataFrame(y_pred5).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())


answer=pd.DataFrame(answer).T.dropna()
total=pd.DataFrame(total).T.dropna()
total['total_ret']=total.sum(axis=1)/5
total['pred1']=y_pred1[:-1]
total['pred2']=y_pred2[:-1]
total['pred3']=y_pred3[:-1]
total['pred4']=y_pred4[:-1]
total['pred5']=y_pred5[:-1]


total['total_ret'].plot()
print("총 누적 수익률은:", ((total['total_ret']+1).cumprod()-1)[-1]*100, "%")
#total[0].plot()
#total[1].plot()
#total[2].plot()
#total[3].plot()
#total[4].plot()

#%%

for i in range(len(total.columns)-6):
    for j in range(len(total[i])-1):
        if i ==0:
            if ((total[i][j] < -0.03) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total[i][j+1] = 0
                
        elif i ==1:
            if ((total[i][j] < -0.03) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total[i][j+1] = 0        
        elif i ==2:
            if ((total[i][j] < -0.03) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total[i][j+1] = 0
        elif i ==3:
            if ((total[i][j] < -0.03) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total[i][j+1] = 0
        elif i ==4:
            if ((total[i][j] < -0.03) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total[i][j+1] = 0
            elif ((total[i][j] == 0) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total[i][j+1] = 0

total['loss_cut']= (total[0]+total[1]+total[2]+total[3]+total[4])/5
((total['loss_cut']+1).cumprod()-1).plot()
         
 #%%
'''  Soft Vector Machine & Gradient Boost & RandomForest 사용'''


def Train_LinearSVM(Xtrain,Xtest,Ytrain,Ytest, C,tol,max_iter):
    
    clf = svm.LinearSVC(penalty='l1', C=C,multi_class='crammer_singer',tol=tol,max_iter=max_iter)
    
    buff=pd.concat([pd.DataFrame(Xtrain), pd.DataFrame(Xtest)])
    
    X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
    X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

    
    clf.fit(X_train,Ytrain)
    
    y_pred = clf.predict(X_test)
    
    pred_df = pd.DataFrame(y_pred)
    print("Accuracy:",metrics.accuracy_score(Ytest, y_pred))   
    conf_matrix=confusion_matrix(Ytest, y_pred)    
    print(conf_matrix)  # -1, 0, 1 순서로 출력한다 
    print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
    print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
    print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
    return pred_df     


def Train_SVM(Xtrain,Xtest,Ytrain,Ytest, C, gamma,decision_function,kernel):
    
    clf = svm.SVC(C=c,gamma=gamma, decision_function_shape = decision_function,kernel = kernel)
    
    buff=pd.concat([pd.DataFrame(Xtrain), pd.DataFrame(Xtest)])
    
    X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
    X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

    
    clf.fit(X_train,Ytrain)
    
    y_pred = clf.predict(X_test)
    
    pred_df = pd.DataFrame(y_pred)

      
    print("Accuracy: {}%".format(np.round(metrics.accuracy_score(Ytest, y_pred),2)*100)) 
    return pred_df



def Train_GradientBoostingCalssifier(Xtrain,Xtest,Ytrain,Ytest,learning_rate,n_estimators):
    
    model = GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators)
    buff=pd.concat([pd.DataFrame(Xtrain), pd.DataFrame(Xtest)])
    
    X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
    X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

    
    model.fit(X_train,Ytrain)
    
    y_pred = model.predict(X_test)
        
    pred_df = pd.DataFrame(y_pred)
    
    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    return pred_df

def Train_RandomForestClassifier(Xtrain,Xtest,Ytrain,Ytest,n_estimators,criterion):
    
    model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion)
    buff=pd.concat([pd.DataFrame(Xtrain), pd.DataFrame(Xtest)])
    
    X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
    X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

    model.fit(X_train, Ytrain)
    
    y_pred = model.predict(X_test)
    
    pred_df = pd.DataFrame(y_pred)
    
    accuracy = accuracy_score(Ytest,y_pred)*100
    
    print("Accuracy: {}%".format(accuracy))
    return pred_df
    

#%% 
    ''' Train Result 확인'''
c=10
gamma = 0.25
decision_function ='ovo'
kernel = 'linear'
tol = 0.0001
max_iter = 5000

print('------------------------------Result of SVM------------------------------')
pred1_SVM = Train_SVM(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred2_SVM = Train_SVM(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred3_SVM = Train_SVM(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred4_SVM = Train_SVM(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,c,gamma,decision_function,kernel)
print('-------------------------------------------------------------------------')
pred5_SVM = Train_SVM(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,c,gamma,decision_function,kernel)

print('---------------------------Result of LinearSVM---------------------------')
pred1_LSVM = Train_LinearSVM(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test, c,tol,max_iter)
print('-------------------------------------------------------------------------')
pred2_LSVM = Train_LinearSVM(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test, c,tol,max_iter)
print('-------------------------------------------------------------------------')
pred3_LSVM = Train_LinearSVM(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test, c,tol,max_iter)
print('-------------------------------------------------------------------------')
pred4_LSVM = Train_LinearSVM(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test, c,tol,max_iter)
print('-------------------------------------------------------------------------')
pred5_LSVM = Train_LinearSVM(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test, c,tol,max_iter)

# Hyper parameters for GradientBoostingClassifier
learning_rate_gb = 0.1
n_estimators_GB = 30


print('------------------------------Result of GBC------------------------------')
pred1_GBC = Train_GradientBoostingCalssifier(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred2_GBC = Train_GradientBoostingCalssifier(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,learning_rate_gb,n_estimators_GB)  
print('-------------------------------------------------------------------------')
pred3_GBC = Train_GradientBoostingCalssifier(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred4_GBC = Train_GradientBoostingCalssifier(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,learning_rate_gb,n_estimators_GB)
print('-------------------------------------------------------------------------')
pred5_GBC = Train_GradientBoostingCalssifier(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,learning_rate_gb,n_estimators_GB)
# Hyper parameters for GradientBoostingClassifier
n_estimators_RF = 30
criterion = 'entropy'


print('------------------------------Result of RF------------------------------')
pred1_RF = Train_RandomForestClassifier(Pair1X_train,Pair1X_test,Pair1Y_train,Pair1Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred2_RF = Train_RandomForestClassifier(Pair2X_train,Pair2X_test,Pair2Y_train,Pair2Y_test,n_estimators_RF,criterion)  
print('-------------------------------------------------------------------------')
pred3_RF = Train_RandomForestClassifier(Pair3X_train,Pair3X_test,Pair3Y_train,Pair3Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred4_RF = Train_RandomForestClassifier(Pair4X_train,Pair4X_test,Pair4Y_train,Pair4Y_test,n_estimators_RF,criterion)
print('-------------------------------------------------------------------------')
pred5_RF = Train_RandomForestClassifier(Pair5X_train,Pair5X_test,Pair5Y_train,Pair5Y_test,n_estimators_RF,criterion)


#%%
''' Signal 합치기 찌릿찌릿'''

Pair1_pred = pd.concat([pred1_SVM,pred1_LSVM,pred1_GBC,pred1_RF],axis=1,join='inner')

Pair1_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair2_pred = pd.concat([pred2_SVM,pred2_LSVM,pred2_GBC,pred2_RF],axis=1,join='inner')

Pair2_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair3_pred = pd.concat([pred3_SVM,pred3_LSVM,pred3_GBC,pred3_RF],axis=1,join='inner')

Pair3_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair4_pred = pd.concat([pred4_SVM,pred4_LSVM,pred4_GBC,pred4_RF],axis=1,join='inner')

Pair4_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']

Pair5_pred = pd.concat([pred5_SVM,pred5_LSVM,pred5_GBC,pred5_RF],axis=1,join='inner')

Pair5_pred.columns = ['pred1_SVM','pred1_LSVM','pred1_GBC','pred1_RF']


def get_voting_score(data):
    
    data['Final_prediction'] = data['pred1_SVM']+data['pred1_LSVM']+\
                                     data['pred1_GBC']+data['pred1_RF']
    
    for i in range(len(Pair1_pred)):
                                
        if data['Final_prediction'][i]== 2 or data['Final_prediction'][i]== -2:
        
           data['Final_prediction'][i]= data['Final_prediction'][i]/2   
        
        elif data['Final_prediction'][i]== 3 or data['Final_prediction'][i]== -3:                               
        
             data['Final_prediction'][i]= data['Final_prediction'][i]/3 
             
        elif data['Final_prediction'][i]== 4 or data['Final_prediction'][i]== -4:
        
             data['Final_prediction'][i]= data['Final_prediction'][i]/4      
             
        elif data['Final_prediction'][i]== 1 or data['Final_prediction'][i]== -1:
        
             data['Final_prediction'][i]= 0   
         
    df = data.values[:,-1]        

    return df

Pair1_pred_vote = get_voting_score(Pair1_pred)
Pair2_pred_vote = get_voting_score(Pair2_pred)
Pair3_pred_vote = get_voting_score(Pair3_pred)
Pair4_pred_vote = get_voting_score(Pair4_pred)
Pair5_pred_vote = get_voting_score(Pair5_pred)

Pairs_pred_vote = [Pair1_pred_vote,Pair2_pred_vote,Pair3_pred_vote,Pair4_pred_vote,Pair5_pred_vote]



#%%



''' 각 자산에 동일한 금액 투자한다고 가정!! '''
''' 한 종목에 100$ 씩 투자한다고 치면, 수익률 계산할 때.. 동일하게 평균내면 됨'''


''' 시그널 여러개일 때 결과 요약'''

def get_cumret(data):
    
    ret = -1*data*(Close[pairs[0][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])+\
             data*(Close[pairs[0][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])
             
    print("누적 수익률은:", (((ret/2)+1).cumprod().iloc[-2]-1)*100,"%")
    print("거래비용은:", (pd.DataFrame(data).replace(-1,1).sum() - (pd.DataFrame(data).replace(0,np.nan) == pd.DataFrame(data).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
    print("Sharpe Ratio:", (((ret/2)).mean())/(ret/2).std())
    return 





payoff=-1*Pair1Y_train*(Close[pairs[0][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Pair1Y_train*(Close[pairs[0][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
((payoff/2)+1).cumprod()
print("학습기간 누적수익률은:", (((payoff/2)+1).cumprod().iloc[-1]-1)*100,"%")
get_cumret(Pair1_pred_vote)


payoff=-1*Pair2Y_train*(Close[pairs[1][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Pair2Y_train*(Close[pairs[1][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
((payoff/2)+1).cumprod()
print("="*60, "\n")
print("학습기간 누적수익률은:", (((payoff/2)+1).cumprod().iloc[-1]-1)*100,"%")
get_cumret(Pair2_pred_vote)


payoff=-1*Pair3Y_train*(Close[pairs[2][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Pair3Y_train*(Close[pairs[2][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
((payoff/2)+1).cumprod()
print("="*60, "\n")
print("학습기간 누적수익률은:", (((payoff/2)+1).cumprod().iloc[-1]-1)*100,"%")
get_cumret(Pair3_pred_vote)




payoff=-1*Pair4Y_train*(Close[pairs[3][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Pair4Y_train*(Close[pairs[3][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
((payoff/2)+1).cumprod()

print("="*60, "\n")
print("학습기간 누적수익률은:", (((payoff/2)+1).cumprod().iloc[-1]-1)*100,"%")
get_cumret(Pair4_pred_vote)




payoff=-1*Pair5Y_train*(Close[pairs[4][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Pair5Y_train*(Close[pairs[4][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
((payoff/2)+1).cumprod()
print("="*60, "\n")
print("학습기간 누적수익률은:", (((payoff/2)+1).cumprod().iloc[-1]-1)*100,"%")
get_cumret(Pair5_pred_vote)



#%% 











for i in range(len(pairs)):
    d_A = model_close[pairs[i][0]]
    d_B= model_close[pairs[i][1]]
    Y=d_A
    X=d_B
    result=reg_m(Y,X)
    beta_.append(result.params[0])
    print(result.summary())  #Durbin - Watson은 에러텀의 독립성.. 시계열 데이터 분석할 때 쓰지.. 
    #첫 페어에서는 2에 가까웠기에 AR(1) 모형에서의 beta값이 조금 그럼..
    # beta의 유의성이 뛰어남!! 
    X_t=result.resid
    spread=Close[pairs[i][0]]-Close[pairs[i][1]]
    
    resid=result.resid
    resid_1=resid.shift(1).dropna()
    resid=resid.iloc[1:]
    result=reg_m(resid, resid_1)
    print(result.summary())
    #AR(1) 모형으로 했는데 Durbin-Watson은 2에 가까움. 추정 잘 됨. beta값도 marginal하게 유의함!(마지막 페어 빼고)
    # 여기서의 x1(=beta)는 theta를 의미함. 즉 평균회귀속도 const=theta*mean 인데.. 거의 0이지.. long-term mean도 거의 0으로 봐도..
    const=result.params[1]
    beta=result.params[0]
    error=result.resid
    
    mu=const/(1-beta)
    sigma= np.sqrt( error.var()/(1-(beta**2)) )
#    
#    buy_open= (pd.DataFrame(X_t >0) * pd.DataFrame(X_t.shift(-1) < X_t))*1
#    sell_open= pd.DataFrame(X_t <0) * pd.DataFrame(X_t.shift(-1) > X_t)*-1
    buy_open= pd.DataFrame(X_t > 1*X_t.std())*1 #* pd.DataFrame(X_t.shift(-1) <1* X_t)*1
    print(buy_open.max())
    sell_open= pd.DataFrame(X_t < -1*X_t.std())*-1 #* pd.DataFrame(X_t.shift(-1) > 1*X_t)*-1
#    for j in range(100):
#        buy_close=((buy_open+sell_open)==0) * (pd.DataFrame(X_t > X_t.mean())*1
#        sell_close=((buy_open+sell_open)==0)*(sell_open.shift(1)==-2) * pd.DataFrame(X_t == X_t.mean())*-1  
    for j in range(100):        
        buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * pd.DataFrame(X_t > X_t.mean()-0.05*X_t.std())*1
        sell_open-=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * pd.DataFrame(X_t < X_t.mean()+ 0.05*X_t.std())*1  
##    sell_open-=(sell_open==0)* (sell_open.shift(1)==-1) * pd.DataFrame(X_t < X_t.mean())
    print(sell_open.min())
    
#    setattr(mod, 'T_price_{}'.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] )
    setattr(mod, 'T_price_{}'.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] )
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) < 0.5* abs(X_t))*2-1).iloc[1:] )
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <  abs(X_t))*2-1).iloc[1:] )
    setattr(mod, 'label_{}'.format(i), pd.DataFrame( buy_open+sell_open+buy_close+sell_close).iloc[1:] ) 
    empty.append(X_t)
#    setattr(mod, 'label_{}'.format(i), pd.DataFrame( (spread.shift(-1) < spread)    *2-1).iloc[1:] )







import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS

os.chdir('C:\\Users\\우람\\Desktop\\kaist\\4차학기\\통차')

data = pd.read_csv('close_new.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace = True)
tickers = list(data.columns.values)
data = data/data.iloc[0]
result_dict = {}
trading_data = data.loc[data.index > '2017-09-08']
data = data.loc[data.index < '2017-09-09']
potential_pairs = pd.read_csv('woooo.csv', index_col = 0)
adf = {}

def half_life(ts):
    ts = np.asarray(ts)
    delta_ts = np.diff(ts)
    lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
    beta = np.linalg.lstsq(lag_ts, delta_ts, rcond = -1)
    return (np.log(2) / beta[0])[0]

for j in range(len(potential_pairs)):
        first = potential_pairs.iloc[j]['first']
        second = potential_pairs.iloc[j]['second']
        pearson = potential_pairs.iloc[j]['Pearson']
        t = adfuller(data[first] - data[second])
        hl = half_life(data[first] - data[second])
        nt = adfuller(trading_data[first] - trading_data[second])
        nhl = half_life(trading_data[first] - trading_data[second])
        adf[potential_pairs.index.values[j]] = [t[0], t[1], hl, pearson, nt[0], nt[1], nhl]

adf_result = pd.DataFrame.from_dict(adf, orient = 'index', 
columns = ['Test Statistic', 'p-value', 'half-life', 'pearson', 'n1', 'n2','n3'])

adf_result = adf_result[adf_result['p-value'] < 0.02]
adf_result[['Test Statistic','p-value','half-life','pearson']].sort_values('p-value')


