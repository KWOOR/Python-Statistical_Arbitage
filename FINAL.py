# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:40:35 2018

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import sys
import statsmodels.api as sm
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pykalman import KalmanFilter
import pandas_datareader.data as web
import datetime
#secs=['EWA', 'EWC']
#data = web.DataReader(secs, 'yahoo', '2010-1-1', '2014-8-1')['Adj Close']


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

''' 기간 나눠서.. 테스트 기간, 트레인 기간 진행!!'''


returns = Close.pct_change()
returns = returns.iloc[1:,:]

#returns_copy=returns

'''Formation Period를 전체의 80%로 둔다'''

returns=returns.iloc[:int(len(returns)*0.8),:]  #Formation Period 동안의 수익률로 페어 찾고 분석 
returns=returns.dropna(axis=1)


''' EigenVector (Loading)을 반환 '''
N_PRIN_COMPONENTS = 40
pca = PCA(n_components=N_PRIN_COMPONENTS)
pca.fit(returns)

pca.components_.T.shape

X=np.array(pca.components_.T)
X = preprocessing.StandardScaler().fit_transform(X)


'''Loading이 같은 종목은 방향성이 같으므로, 같은 종목들끼리만 분류하기'''
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


    
which_cluster = clustered_series.loc['DUK UN Equity']  
clustered_series[clustered_series == which_cluster]


tickers = list(clustered_series[clustered_series==which_cluster].index)
means = np.log(Close[tickers].mean())
data = np.log(Close[tickers]).sub(means)
data.plot(legend=False, title="Stock Time Series for Cluster %d" % which_cluster);

''' DBSCAN을 통한 후보군 중에서 Cointegrate 5%수준으로 Mean-reverting 특성이 있는 최종 5개 페어 고르기 '''

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


pairs=[('HON UN Equity', 'MMC UN Equity'), ('HON UN Equity', 'AON UN Equity'), 
       ('BBT UN Equity', 'STI UN Equity'), ('DUK UN Equity', 'SRE UN Equity'), 
       ('LMT UN Equity', 'NOC UN Equity')]  #혹시 몰라서 5개 페어 저장..

for i in range(len(pairs)):
    Close[list(pairs[i])].plot()
    ax = (Close[pairs[i][0]]-Close[pairs[i][1]]).plot(title='Stock price of pairs and spread')
    ax.legend(['{}'.format(pairs[i][0]),'{}'.format(pairs[i][1]),'Spread'])

print("Pair Selection 완료")

del CLUSTER_SIZE_LIMIT, KMeans, TSNE, DBSCAN,PCA,clf, clust, cluster_dict, cluster_vis_list, clustered_series_all, data,find_cointegrated_pairs, which_clust,which_cluster,ticker_count,ticker_count_reduced,n_clusters_,pca
#%%
''' Data Set 만들기 '''

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

'''T_Score 계산'''
                
def get_Tscore(data1,data2, status, absvalue=-1, Kalman=False): #data1은 종가, data2는 시가, status는 'T_price_{}' 이런식으로!!
    #status=T_price_{}, T_wma_{}, T_sma_{}, T_rsi_{}, T_mfi_{} string형식으로..
    for i in range(len(pairs)):
        d_A = data1[pairs[i][0]]-data2[pairs[i][0]]
        d_B=  data1[pairs[i][1]]-data2[pairs[i][1]]
        Y= (d_A/data2[pairs[i][0]]).dropna()
        X= (d_B/data2[pairs[i][1]]).dropna()
        result=reg_m(Y,X)
        X_t=result.resid
        
        resid=result.resid
        resid_1=resid.shift(1).dropna()
        resid=resid.iloc[1:]
        
        if Kalman==True:
            obs_mat = np.vstack([resid_1, np.ones(resid_1.shape)]).T[:, np.newaxis]
            delta = 1e-5
            trans_cov = delta / (1 - delta) * np.eye(2)
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                              initial_state_mean=np.zeros(2),
                              initial_state_covariance=np.ones((2, 2)),
                              transition_matrices=np.eye(2),
                              observation_matrices=obs_mat,
                              observation_covariance=1.0,
                              transition_covariance=trans_cov)
            state_means, state_covs = kf.filter(resid.values) #means 0이 slope, 1이 constant
            beta=state_means[:,0]
            const=state_means[:,1]
            error=resid- (resid_1*beta) -const
            mu=const/(1-beta)
            mu=np.hstack([mu[0], mu])
            sigma= np.sqrt( error.var()/(1-(beta**2)) )
            sigma=np.hstack([sigma[0], sigma])
            
        else:
            result=reg_m(resid, resid_1)
            const=result.params[1]
            beta=result.params[0]
            error=result.resid
            mu=const/(1-beta)
            sigma= np.sqrt( error.var()/(1-(beta**2)) )
        
        if absvalue==-1:  #방향성 맞추기
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )).iloc[1:] ), 
                setattr(mod, 'label_{}'.format(i), pd.DataFrame((abs(X_t.shift(-1)) <  abs(X_t))*2-1).iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( abs ( (X_t - mu) /sigma )) )
                
        elif absvalue==0: #PCA에 있던 ppt Score방식 사용
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] ) 
                buy_open= pd.DataFrame( ((X_t - mu) /sigma) > 1.25 )*1
                sell_open= pd.DataFrame( ((X_t - mu) /sigma) < -1.25 )*-1
                buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * (pd.DataFrame( ((X_t-mu)/sigma) > 0.5))*1
                sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * (pd.DataFrame( ((X_t-mu)/sigma) <-0.75))*-1  

                setattr(mod, 'label_{}'.format(i), pd.DataFrame(buy_open+sell_open,dtype='i').iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( (X_t - mu) /sigma ) )
                
        else: #거래 Signal 맞추기 
            if status=='T_price_{}':
                setattr(mod, status.format(i), pd.DataFrame(  (X_t - mu) /sigma ).iloc[1:] ) 
                buy_open= (pd.DataFrame(X_t > 0) & pd.DataFrame(X_t.shift(-1) < absvalue*X_t))*1
                sell_open= (pd.DataFrame(X_t < 0) & pd.DataFrame(X_t.shift(-1) > absvalue*X_t))*-1
                buy_open+=((buy_open+sell_open)==0)*(buy_open.shift(1)==1) * (pd.DataFrame(X_t > 0))*1
                sell_open+=((buy_open+sell_open)==0)*(sell_open.shift(1)==-1) * (pd.DataFrame(X_t <0))*-1  

                setattr(mod, 'label_{}'.format(i), pd.DataFrame(buy_open+sell_open,dtype='i').iloc[1:] )
            else:
                setattr(mod, status.format(i), pd.DataFrame( (X_t - mu) /sigma ) )
                
                
value=-1  #-1이면 방향성 맞추기, 0이면 거래 Signal 중에서 PCA ppt에 있던 S-score 방식 사용하기
kal=True #False면 Kalman Filter 안 씀 
get_Tscore(model_close, model_open, 'T_price_{}', value, kal)
get_Tscore(model_wma_close, model_wma_open, 'T_wma_{}', value, kal)
get_Tscore(model_sma_close, model_sma_open, 'T_sma_{}', value, kal)
get_Tscore(model_rsi_close, model_rsi_open, 'T_rsi_{}', value, kal)
get_Tscore(model_mfi_close, model_mfi_open, 'T_mfi_{}', value, kal)


''' Data Concat 및 학습, 테스트 데이터로 분류하기 '''
criteria=0.8 #80%가 트레인셋, 20%가 테스트셋
l=locals()
for i in range(len(pairs)):
    setattr(mod, 'Pair{}'.format(i+1), pd.concat([l['T_price_%d'%i],l['T_mfi_%d'%i],
            l['T_rsi_%d'%i],l['T_sma_%d'%i], l['T_wma_%d'%i], l['label_%d'%i]], axis=1, 
        join='inner').values)        
    
    setattr(mod, 'Pair{}X_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),:-1])
    setattr(mod, 'Pair{}X_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,:-1])
    setattr(mod, 'Pair{}Y_train'.format(i+1), l['Pair%d'%(i+1)][:int(len(l['Pair%d'%(i+1)])*criteria),-1])
    setattr(mod, 'Pair{}Y_test'.format(i+1), l['Pair%d'%(i+1)][int(len(l['Pair%d'%(i+1)])*criteria):,-1])

#del  Open, Sma,Wma, Rsi, Mfi

#%%
'''Soft Vector Machine 적용'''
def MinMaxScale(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def Train_LinearSVM(Xtrain,Xtest,Ytrain,Ytest, C=10,tol=0.0001,max_iter=1000):
    
    clf = svm.LinearSVC( C=C,tol=tol,max_iter=max_iter)
    
    buff=pd.concat([pd.DataFrame(Xtrain), pd.DataFrame(Xtest)])
    
    X_train=np.array(MinMaxScale(buff)[:int(len(buff)*0.8)])
    X_test=np.array(MinMaxScale(buff)[int(len(buff)*0.8):])

    
    clf.fit(X_train,Ytrain)
    
    y_pred = clf.predict(X_test)
    
    pred_df = y_pred
    print("Accuracy:",metrics.accuracy_score(Ytest, y_pred))   
    conf_matrix=confusion_matrix(Ytest, y_pred)    
    print(conf_matrix)  # -1, 0, 1 순서로 출력한다 
    print(classification_report(Ytest, y_pred))
#    print("Recall:", (conf_matrix[2][2]+conf_matrix[0][0])/ (conf_matrix[2][2]+ conf_matrix[2][0]+conf_matrix[2][1]+conf_matrix[0][0]+ conf_matrix[0][1]+conf_matrix[0][2])) #성공적으로 이익 기회를 잡음
#    print("Opposite Recall:", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]+conf_matrix[0][2])) #손실 기회를 성공적으로 피함
#    print("Precision:", (conf_matrix[2][2]+conf_matrix[0][0]) / (conf_matrix[2][2] + conf_matrix[1][2]+conf_matrix[0][2]+conf_matrix[0][0]+conf_matrix[1][0]+conf_matrix[2][0])) # 전체 거래 횟수 중에 이익을 본 거래 횟수
    return pred_df     



y_pred1=Train_LinearSVM(Pair1X_train, Pair1X_test, Pair1Y_train, Pair1Y_test)
y_pred2=Train_LinearSVM(Pair2X_train, Pair2X_test, Pair2Y_train, Pair2Y_test)
y_pred3=Train_LinearSVM(Pair3X_train, Pair3X_test, Pair3Y_train, Pair3Y_test)
y_pred4=Train_LinearSVM(Pair4X_train, Pair4X_test, Pair4Y_train, Pair4Y_test)
y_pred5=Train_LinearSVM(Pair5X_train, Pair5X_test, Pair5Y_train, Pair5Y_test)


''' 수익률 계산하기 '''
def get_ret(Y_train, Y_test, y_pred, num_of_pairs):
    answer=[]
    total=[]
    payoff=-1*Y_train*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])+Y_train*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_mfi_0.index][:int(len(T_mfi_0)*0.8)])
    print("Training 기간 동안 누적 수익률은:",(((payoff)+1).cumprod().iloc[-1]-1)*100,"%")
    payoff=-1*Y_test*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])+Y_test*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])
    answer.append(((payoff)+1).cumprod())
    print("Test기간 다 맞췄을 때, 누적수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
    payoff=-1*y_pred*(Close[pairs[num_of_pairs][0]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])+y_pred*(Close[pairs[num_of_pairs][1]].pct_change().shift(-1).loc[T_mfi_0.index][int(len(T_mfi_0)*0.8):])
    total.append(((payoff)))
    print("실제 거래 누적 수익률은:", (((payoff)+1).cumprod().iloc[-2]-1)*100,"%")
    print("거래비용은:", (pd.DataFrame(y_pred).replace(-1,1).sum() - (pd.DataFrame(y_pred).replace(0,np.nan) == pd.DataFrame(y_pred).replace(0,np.nan).shift(1)).sum()).values*0.3,"%" )
    print("Sharpe Ratio:", (((payoff)).mean())/(payoff).std())
    print("="*60, "\n")
    return pd.DataFrame(answer), pd.DataFrame(total)

answer1, total1 =get_ret(Pair1Y_train, Pair1Y_test, y_pred1, 0)
answer2, total2 =get_ret(Pair2Y_train, Pair2Y_test, y_pred2, 1)
answer3, total3 =get_ret(Pair3Y_train, Pair3Y_test, y_pred3, 2)
answer4, total4 =get_ret(Pair4Y_train, Pair4Y_test, y_pred4, 3)
answer5, total5 =get_ret(Pair5Y_train, Pair5Y_test, y_pred5, 4)


answer=pd.concat([answer1.T,answer2.T,answer3.T,answer4.T,answer5.T], axis=1).dropna()
total=pd.concat([total1.T,total2.T,total3.T,total4.T,total5.T], axis=1).dropna()

total['total_ret']=total.sum(axis=1)/5
total['pred1']=y_pred1[:-1]
total['pred2']=y_pred2[:-1]
total['pred3']=y_pred3[:-1]
total['pred4']=y_pred4[:-1]
total['pred5']=y_pred5[:-1]


#total['total_ret'].plot()
print("총 누적 수익률은:", ((total['total_ret']+1).cumprod()-1)[-1]*100, "%")

#%%
''' Loss Cut 적용했을 때'''
lc=-0.01 # 1% Loss Cut
for i in range(len(total.columns)-6):
    for j in range(len(total)-1):
        if i ==0:
            if ((total.iloc[j,i] < lc) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total.iloc[j+1,i] = 0
            elif ((total.iloc[j,i] == 0) * (total['pred1'][j]==total['pred1'][j-1]) *( total['pred1'][j]==total['pred1'][j+1])) == True:
                total.iloc[j+1,i] = 0
                
        elif i ==1:
            if ((total.iloc[j,i] < lc) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total.iloc[j+1,i] = 0
            elif ((total.iloc[j,i] == 0) * (total['pred2'][j]==total['pred2'][j-1]) *( total['pred2'][j]==total['pred2'][j+1])) == True:
                total.iloc[j+1,i] = 0        
        elif i ==2:
            if ((total.iloc[j,i] < lc) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total.iloc[j+1,i] = 0
            elif ((total.iloc[j,i] == 0) * (total['pred3'][j]==total['pred3'][j-1]) *( total['pred3'][j]==total['pred3'][j+1])) == True:
                total.iloc[j+1,i] = 0
        elif i ==3:
            if ((total.iloc[j,i] < lc) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total.iloc[j+1,i] = 0
            elif ((total.iloc[j,i] == 0) * (total['pred4'][j]==total['pred4'][j-1]) *( total['pred4'][j]==total['pred4'][j+1])) == True:
                total.iloc[j+1,i] = 0
        elif i ==4:
            if ((total.iloc[j,i] < lc) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total.iloc[j+1,i] = 0
            elif ((total.iloc[j,i] == 0) * (total['pred5'][j]==total['pred5'][j-1]) *( total['pred5'][j]==total['pred5'][j+1])) == True:
                total.iloc[j+1,i] = 0

total['loss_cut']= (total[0].sum(axis=1))/5
((total['loss_cut']+1).cumprod()-1).plot()
print("Loss Cut 적용한 총 누적 수익률은:", ((total['loss_cut']+1).cumprod()-1)[-1]*100, "%")
((total[0]+1).cumprod()-1).iloc[-1].sum()/5



#%%

#%%
''' Distance Approach '''
data = Close/Close.iloc[0]
trading_data = data.loc[data.index >= '2017-09-14']
formation_data=data['2013-01-24':'2017-09-13']
pairs=np.array(pairs)

def trading_signals(first, second, trading_data = trading_data, formation_data = formation_data):
    #choose 2-sigma as the trading signal
    signal = 2*np.std(formation_data[first] - formation_data[second])
    result_dict = {}
    
    #there should be no trading initially
    trading = False
    
    #create a time series of the spread between the two stocks
    differences = trading_data[first] - trading_data[second]
    for i in range(len(differences)):
        
        #if there is no trading, OPEN it if the spread is greater than the signal
        #AND the spread is less than the stop-loss of 4-sigma
        #if not, move onto the next day
        if trading == False:
            if abs(differences.iloc[i]) > signal and abs(differences.iloc[i] < 2*signal):
                trading = True
                start_date = differences.index.values[i]
                
        #if the trade is already open, we check to see if the spread has crossed OR exceeded the 4-sigma stoploss
        #we close the trade and record the start and end date of the trade
        #we also record the return from the short and long position of the trade
        else:
            if (differences.iloc[i-1] * differences.iloc[i] < 0) or (i == len(differences)-1) or abs(differences.iloc[i] > 2*signal):
                trading = False
                end_date = differences.index.values[i]
                if differences[i-1] > 0:
                    s_ret = (trading_data[first][start_date] - trading_data[first][end_date])/trading_data[first][start_date]
                    l_ret = (trading_data[second][end_date] - trading_data[second][start_date])/trading_data[second][start_date]
                    result_dict[start_date] = [first, second, start_date, end_date, s_ret,l_ret]
                else:
                    s_ret = (trading_data[second][start_date] - trading_data[second][end_date])/trading_data[second][start_date]
                    l_ret = (trading_data[first][end_date] - trading_data[first][start_date])/trading_data[first][start_date]
                    result_dict[start_date] = [second, first, start_date, end_date, s_ret,l_ret]
    
    #formatting the final dataframe to be returned
    df = pd.DataFrame.from_dict(result_dict, orient = 'index', columns = ['Short','Long','Start','End', 'SReturn','LReturn'])
    df.index = list(range(len(df)))
    df['Total'] = df['SReturn'] + df['LReturn']
    df['Length'] = (df['End'] - df['Start']).dt.days
    return (df, len(df))

trade_cost=0.003
def build_portfolio(trade_list, trading_data = trading_data, trade_cost=0.003):
    #create a index_list of dates
    index_list = trading_data.index.tolist()
    
    #initialize dataframe
    portfolio = pd.DataFrame(index = trading_data.index.values, columns = ['Short','Long','ShortR','LongR','Trading'])
    l = trade_list[1]
    trade_list = trade_list[0]
    
    #for each trade, find the start and end dates, and which stocks to long/short
    for i in range(len(trade_list)):
        start = trade_list['Start'][i]
        end = trade_list['End'][i]
        short = trade_list['Short'][i]
        lon = trade_list['Long'][i]
        di = index_list.index(start)
        di2 = index_list.index(end)
        
        #from the start to end date, add the value of the position from that day for that stock
        #also take away trade cost (for long) or add it for shorts
        for j in range(di2 - di + 1):
            date_index = di + j
            dt = index_list[date_index]
            portfolio['Short'][dt] = (trading_data[short][dt]/trading_data[short][index_list[di]]) + trade_cost
            portfolio['Long'][dt] = trading_data[lon][dt]/trading_data[lon][index_list[di]] - trade_cost
            portfolio['Trading'][dt] = 1
            if j == (di2 - di):
                portfolio['Short'][dt] = portfolio['Short'][dt] + trade_cost
                portfolio['Long'][dt] = portfolio['Long'][dt] - trade_cost

    #fill non-trading days
    portfolio.fillna(value = 0, axis = 0)
      
    #adding columns for returns from the short and long portions of the portfolio
    for j in range(1, len(portfolio)):
        if portfolio.iloc[j-1]['Short'] > 0:
            portfolio.iloc[j]['ShortR'] = -(portfolio.iloc[j]['Short'] - portfolio.iloc[j-1]['Short'])/portfolio.iloc[j-1]['Short']
            portfolio.iloc[j]['LongR'] = (portfolio.iloc[j]['Long'] - portfolio.iloc[j-1]['Long'])/portfolio.iloc[j-1]['Long']
        else:
            portfolio.iloc[j]['ShortR'] = 0
            portfolio.iloc[j]['LongR']= 0
            
    #total return is teh sum of both returns
    portfolio['Total'] = portfolio['ShortR'] + portfolio['LongR']
    portfolio.fillna(0, inplace = True)
    return (portfolio, l)


def analyze_portfolio(pairs):
    i = 0
    df = (build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[0])
    trade_count = build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
    for i in range(1, len(pairs)):
        df = df + (build_portfolio(trading_signals(pairs[i][0], pairs[i][1])))[0]
        trade_count += build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
    df_short = df['ShortR']/5
    df_long = df['LongR']/5
    df_final = pd.concat([df_short, df_long], axis=1)
    df_final.columns = ['Short Return','Long Return']
    df_final.index.name = 'Date'
    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
    df_final.fillna(0, inplace = True)
    arithemtic_daily_mean = np.mean(df_final['Total'])
    annualized_return = (1+arithemtic_daily_mean)**250 - 1
    annualized_std = np.std(df_final['Total'])*np.sqrt(250)
    sharpe_ratio = annualized_return/annualized_std
    return [annualized_return, annualized_std, sharpe_ratio, trade_count]



for i in range(len(pairs)):
    empty=trading_signals(pairs[i][0], pairs[i][1])[0]
    print("{}번째 Pair 총 거래 횟수는".format(i) , len(empty), "번 이다.")
    buff=build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[0]
    print("{}번째 Pair 누적 수익률은".format(i), ((buff['Total']+1).cumprod()[-1]-1)*100, "%"  )
a_r, a_std, sh, tc = analyze_portfolio(pairs)
print("총 포트폴리오의 연간 수익률은" , a_r*100, "%")
print("총 포트폴리오의 연간 변동성은" , a_std*100, "%")
print("총 포트폴리오의 Shapre Ratio는" , sh)
print("총 포트폴리오 거래 횟수는" , tc)

i = 0
df = (build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[0])
trade_count = build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
for i in range(1, len(pairs)):
    df = df + (build_portfolio(trading_signals(pairs[i][0], pairs[i][1])))[0]
    trade_count += build_portfolio(trading_signals(pairs[i][0], pairs[i][1]))[1]
df_short = df['ShortR']/5
df_long = df['LongR']/5
df_final = pd.concat([df_short, df_long], axis=1)
df_final.columns = ['Short Return','Long Return']
df_final.index.name = 'Date'
df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
df_final.fillna(0, inplace = True)
    
((df_final['Total']+1).cumprod()-1).plot()
    
    
#%%

''' DA + ML'''

def plus_trading_signals(first, second, pred, trading_data = trading_data, formation_data = formation_data):
    #choose 2-sigma as the trading signal
    signal = 2*np.std(formation_data[first] - formation_data[second])
    result_dict = {}
    
    #there should be no trading initially
    trading = False
    
    #create a time series of the spread between the two stocks
    differences = trading_data[first] - trading_data[second]
    for i in range(len(differences)):
        
        #if there is no trading, OPEN it if the spread is greater than the signal
        #AND the spread is less than the stop-loss of 4-sigma
        #if not, move onto the next day
        if trading == False:
            if abs(differences.iloc[i]) > signal and abs(differences.iloc[i] < 2*signal) and (pred[i] ==1):
                trading = True
                start_date = differences.index.values[i]
                
        #if the trade is already open, we check to see if the spread has crossed OR exceeded the 4-sigma stoploss
        #we close the trade and record the start and end date of the trade
        #we also record the return from the short and long position of the trade
        else:
            if (differences.iloc[i-1] * differences.iloc[i] < 0) or (i == len(differences)-1) or abs(differences.iloc[i] > 2*signal):
                trading = False
                end_date = differences.index.values[i]
                if differences[i-1] > 0:
                    s_ret = (trading_data[first][start_date] - trading_data[first][end_date])/trading_data[first][start_date]
                    l_ret = (trading_data[second][end_date] - trading_data[second][start_date])/trading_data[second][start_date]
                    result_dict[start_date] = [first, second, start_date, end_date, s_ret,l_ret]
                else:
                    s_ret = (trading_data[second][start_date] - trading_data[second][end_date])/trading_data[second][start_date]
                    l_ret = (trading_data[first][end_date] - trading_data[first][start_date])/trading_data[first][start_date]
                    result_dict[start_date] = [second, first, start_date, end_date, s_ret,l_ret]
    
    #formatting the final dataframe to be returned
    df = pd.DataFrame.from_dict(result_dict, orient = 'index', columns = ['Short','Long','Start','End', 'SReturn','LReturn'])
    df.index = list(range(len(df)))
    df['Total'] = df['SReturn'] + df['LReturn']
    df['Length'] = (df['End'] - df['Start']).dt.days
    return (df, len(df))

trade_cost=0.003
def plus_build_portfolio(trade_list, trading_data = trading_data, trade_cost=0.003):
    #create a index_list of dates
    index_list = trading_data.index.tolist()
    
    #initialize dataframe
    portfolio = pd.DataFrame(index = trading_data.index.values, columns = ['Short','Long','ShortR','LongR','Trading'])
    l = trade_list[1]
    trade_list = trade_list[0]
    
    #for each trade, find the start and end dates, and which stocks to long/short
    for i in range(len(trade_list)):
        start = trade_list['Start'][i]
        end = trade_list['End'][i]
        short = trade_list['Short'][i]
        lon = trade_list['Long'][i]
        di = index_list.index(start)
        di2 = index_list.index(end)
        
        #from the start to end date, add the value of the position from that day for that stock
        #also take away trade cost (for long) or add it for shorts
        for j in range(di2 - di + 1):
            date_index = di + j
            dt = index_list[date_index]
            portfolio['Short'][dt] = (trading_data[short][dt]/trading_data[short][index_list[di]]) + trade_cost
            portfolio['Long'][dt] = trading_data[lon][dt]/trading_data[lon][index_list[di]] - trade_cost
            portfolio['Trading'][dt] = 1
            if j == (di2 - di):
                portfolio['Short'][dt] = portfolio['Short'][dt] + trade_cost
                portfolio['Long'][dt] = portfolio['Long'][dt] - trade_cost

    #fill non-trading days
    portfolio.fillna(value = 0, axis = 0)
      
    #adding columns for returns from the short and long portions of the portfolio
    for j in range(1, len(portfolio)):
        if portfolio.iloc[j-1]['Short'] > 0:
            portfolio.iloc[j]['ShortR'] = -(portfolio.iloc[j]['Short'] - portfolio.iloc[j-1]['Short'])/portfolio.iloc[j-1]['Short']
            portfolio.iloc[j]['LongR'] = (portfolio.iloc[j]['Long'] - portfolio.iloc[j-1]['Long'])/portfolio.iloc[j-1]['Long']
        else:
            portfolio.iloc[j]['ShortR'] = 0
            portfolio.iloc[j]['LongR']= 0
            
    #total return is teh sum of both returns
    portfolio['Total'] = portfolio['ShortR'] + portfolio['LongR']
    portfolio.fillna(0, inplace = True)
    return (portfolio, l)


def plus_analyze_portfolio(pairs):
    i = 0
    df = (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0])
    trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
    for i in range(1, len(pairs)):
        df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)])))[0]
        trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],  l['y_pred%d'%(i+1)]))[1]
    df_short = df['ShortR']/5
    df_long = df['LongR']/5
    df_final = pd.concat([df_short, df_long], axis=1)
    df_final.columns = ['Short Return','Long Return']
    df_final.index.name = 'Date'
    df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
    df_final.fillna(0, inplace = True)
    arithemtic_daily_mean = np.mean(df_final['Total'])
    annualized_return = (1+arithemtic_daily_mean)**250 - 1
    annualized_std = np.std(df_final['Total'])*np.sqrt(250)
    sharpe_ratio = annualized_return/annualized_std
    return [annualized_return, annualized_std, sharpe_ratio, trade_count]


for i in range(len(pairs)):
    empty1=plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)])[0]
    print("{}번째 Pair 총 거래 횟수는".format(i) , len(empty1), "번 이다.")
    buff1=plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0]
    print("{}번째 Pair 누적 수익률은".format(i), ((buff1['Total']+1).cumprod()[-1]-1)*100, "%"  )
    
a_r, a_std, sh, tc = plus_analyze_portfolio(pairs)
print("총 포트폴리오의 연간 수익률은" , a_r*100, "%")
print("총 포트폴리오의 연간 변동성은" , a_std*100, "%")
print("총 포트폴리오의 Shapre Ratio는" , sh)
print("총 포트폴리오 거래 횟수는" , tc)

i = 0
df = (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[0])
trade_count = plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)]))[1]
for i in range(1, len(pairs)):
    df = df + (plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1], l['y_pred%d'%(i+1)])))[0]
    trade_count += plus_build_portfolio(plus_trading_signals(pairs[i][0], pairs[i][1],l['y_pred%d'%(i+1)]))[1]
df_short = df['ShortR']/5
df_long = df['LongR']/5
df_final = pd.concat([df_short, df_long], axis=1)
df_final.columns = ['Short Return','Long Return']
df_final.index.name = 'Date'
df_final['Total'] = df_final['Short Return'] + df_final['Long Return']
df_final.fillna(0, inplace = True)
    
((df_final['Total']+1).cumprod()-1).plot()
    













