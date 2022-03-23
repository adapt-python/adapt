#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:23:55 2022

@author: mounir
"""

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from adapt.parameter_based import TransferTreeClassifier, TransferForestClassifier

methods = [
    'relab',
    'ser',
    'strut',
    'ser_nr',
    'ser_nr_lambda',
    'strut_nd',
    'strut_lambda',
    'strut_lambda_np'
#    'strut_hi'
]
labels = [
    'relab',
    '$SER$',
    '$STRUT$',
    '$SER_{NP}$',
    '$SER_{NP}(\lambda)$',
    '$STRUT_{ND}$',
    '$STRUT(\lambda)$',
    '$STRUT_{NP}(\lambda)$'
    # 'STRUT$^{*}$',
    #'STRUT$^{*}$',
]

DT_only = False
RF_SIZE = 10


np.random.seed(0)

plot_step = 0.01
# Generate training source data
ns = 200
ns_perclass = ns // 2
mean_1 = (1, 1)
var_1 = np.diag([1, 1])
mean_2 = (3, 3)
var_2 = np.diag([2, 2])
Xs = np.r_[np.random.multivariate_normal(mean_1, var_1, size=ns_perclass),
           np.random.multivariate_normal(mean_2, var_2, size=ns_perclass)]
ys = np.zeros(ns)
ys[ns_perclass:] = 1
# Generate training target data
nt = 50
# imbalanced
nt_0 = nt // 10
mean_1 = (6, 3)
var_1 = np.diag([4, 1])
mean_2 = (5, 5)
var_2 = np.diag([1, 3])
Xt = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_0),
           np.random.multivariate_normal(mean_2, var_2, size=nt - nt_0)]
yt = np.zeros(nt)
yt[nt_0:] = 1
# Generate testing target data
nt_test = 1000
nt_test_perclass = nt_test // 2
Xt_test = np.r_[np.random.multivariate_normal(mean_1, var_1, size=nt_test_perclass),
                np.random.multivariate_normal(mean_2, var_2, size=nt_test_perclass)]
yt_test = np.zeros(nt_test)
yt_test[nt_test_perclass:] = 1

# Source classifier
if DT_only:
    clf_source = DecisionTreeClassifier(max_depth=None)
else:
    clf_source = RandomForestClassifier(n_estimators=RF_SIZE)
    
clf_source.fit(Xs, ys)
score_src_src = clf_source.score(Xs, ys)
score_src_trgt = clf_source.score(Xt_test, yt_test)
print('Training score Source model: {:.3f}'.format(score_src_src))
print('Testing score Source model: {:.3f}'.format(score_src_trgt))
clfs = []
scores = []
# Transfer with SER
#clf_transfer = copy.deepcopy(clf_source)
#transferred_dt = TL.TransferTreeClassifier(estimator=clf_transfer,Xt=Xt,yt=yt)

for method in methods:
    
    if DT_only:        
        Nkmin = sum(yt == 0 )
        root_source_values = clf_source.tree_.value[0].reshape(-1)
        props_s = root_source_values
        props_s = props_s / sum(props_s)
        props_t = np.zeros(props_s.size)
        for k in range(props_s.size):
            props_t[k] = np.sum(yt == k) / yt.size
        
        coeffs = np.divide(props_t, props_s)          
    
    clf_transfer = copy.deepcopy(clf_source)
    
    if method == 'relab':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="")
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="",bootstrap=False)
        print(transferred_model.algo)
        transferred_model.fit(Xt,yt)
        
    if method == 'ser':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="ser")
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="ser")
        transferred_model.fit(Xt,yt)
        #transferred_dt._ser(Xt, yt, node=0, original_ser=True)
        #ser.SER(0, clf_transfer, Xt, yt, original_ser=True)
    if method == 'ser_nr':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="ser")
            transferred_model._ser(Xt, yt,node=0,original_ser=False,no_red_on_cl=True,cl_no_red=[0])

        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="ser")            
            transferred_model._ser_rf(Xt, yt,original_ser=False,no_red_on_cl=True,cl_no_red=[0])
        
    if method == 'ser_nr_lambda':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="ser")
            transferred_model._ser(Xt, yt,node=0,original_ser=False,no_red_on_cl=True,cl_no_red=[0],
                            leaf_loss_quantify=True,leaf_loss_threshold=0.5,
                            root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="ser")            
            transferred_model._ser_rf(Xt, yt,original_ser=False,no_red_on_cl=True,cl_no_red=[0],
                            leaf_loss_quantify=True,leaf_loss_threshold=0.5)
        #ser.SER(0, clf_transfer, Xt, yt,original_ser=False,no_red_on_cl=True,cl_no_red=[0],ext_cond=True)
        
    if method == 'strut':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="strut")
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="strut")
        transferred_model.fit(Xt,yt)
        #transferred_dt._strut(Xt, yt,node=0)
        
    if method == 'strut_nd':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="strut")
            transferred_model._strut(Xt, yt,node=0,use_divergence=False)
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="strut")            
            transferred_model._strut_rf(Xt, yt,use_divergence=False)
        
    if method == 'strut_lambda':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="strut")
            transferred_model._strut(Xt, yt,node=0,adapt_prop=True,root_source_values=root_source_values,
                              Nkmin=Nkmin,coeffs=coeffs)
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="strut")            
            transferred_model._strut_rf(Xt, yt,adapt_prop=True)
        
    if method == 'strut_lambda_np':
        if DT_only :
            transferred_model = TransferTreeClassifier(estimator=clf_transfer,algo="strut")            
            transferred_model._strut(Xt, yt,node=0,adapt_prop=True,no_prune_on_cl=True,cl_no_prune=[0],
                            leaf_loss_quantify=True,leaf_loss_threshold=0.5,no_prune_with_translation=True,
                            root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
        else:
            transferred_model = TransferForestClassifier(estimator=clf_transfer,algo="strut")            
            transferred_model._strut_rf(Xt, yt,adapt_prop=True,no_prune_on_cl=True,cl_no_prune=[0],
                            leaf_loss_quantify=True,leaf_loss_threshold=0.5,no_prune_with_translation=True)

    #if method == 'strut_hi':
        #transferred_dt._strut(Xt, yt,node=0,no_prune_on_cl=False,adapt_prop=True,coeffs=[0.2, 1])
        #strut.STRUT(clf_transfer, 0, Xt, yt, Xt, yt,pruning_updated_node=True,no_prune_on_cl=False,adapt_prop=True,simple_weights=False,coeffs=[0.2, 1])
    score = transferred_model.estimator.score(Xt_test, yt_test)
    #score = clf_transfer.score(Xt_test, yt_test)
    print('Testing score transferred model ({}) : {:.3f}'.format(method, score))
    clfs.append(transferred_model.estimator)
    #clfs.append(clf_transfer)
    scores.append(score)

# Plot decision functions

# Data on which to plot source
x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
# Plot source model
Z = clf_source.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(nrows=1, ncols=len(methods) + 1, figsize=(30, 3))
ax[0].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
ax[0].scatter(Xs[0, 0], Xs[0, 1],
              marker='o',
              edgecolor='black',
              color='white',
              label='source data',
              )
ax[0].scatter(Xs[:ns_perclass, 0], Xs[:ns_perclass, 1],
              marker='o',
              edgecolor='black',
              color='blue',
              )
ax[0].scatter(Xs[ns_perclass:, 0], Xs[ns_perclass:, 1],
              marker='o',
              edgecolor='black',
              color='red',
              )
ax[0].set_title('Model: Source\nAcc on source data: {:.2f}\nAcc on target data: {:.2f}'.format(score_src_src, score_src_trgt),
                fontsize=11)
ax[0].legend()

# Data on which to plot target
x_min, x_max = Xt[:, 0].min() - 1, Xt[:, 0].max() + 1
y_min, y_max = Xt[:, 1].min() - 1, Xt[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
# Plot transfer models
for i, (method, label, score) in enumerate(zip(methods, labels, scores)):
    clf_transfer = clfs[i]
    Z_transfer = clf_transfer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_transfer = Z_transfer.reshape(xx.shape)
    ax[i + 1].contourf(xx, yy, Z_transfer, cmap=plt.cm.coolwarm, alpha=0.8)
    ax[i + 1].scatter(Xt[0, 0], Xt[0, 1],
                      marker='o',
                      edgecolor='black',
                      color='white',
                      label='target data',
                      )
    ax[i + 1].scatter(Xt[:nt_0, 0], Xt[:nt_0, 1],
                      marker='o',
                      edgecolor='black',
                      color='blue',
                      )
    ax[i + 1].scatter(Xt[nt_0:, 0], Xt[nt_0:, 1],
                      marker='o',
                      edgecolor='black',
                      color='red',
                      )
    ax[i + 1].set_title('Model: {}\nAcc on target data: {:.2f}'.format(label, score),
                        fontsize=11)
    ax[i + 1].legend()

# fig.savefig('../images/ser_strut.png')
plt.show()

