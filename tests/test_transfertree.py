import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from adapt.parameter_based import TransferTreeClassifier, TransferForestClassifier

methods = [
    'relab',
    'ser',
    'strut',
    'ser_nr',
    'ser_no_ext',
    'ser_nr_lambda',
    'strut_nd',
    'strut_lambda',
    'strut_np'
    'strut_lambda_np',
    'strut_lambda_np2'
#    'strut_hi'
]


def test_transfer_tree():

    np.random.seed(0)

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
    RF_SIZE = 10
    clf_source_dt = DecisionTreeClassifier(max_depth=None)
    clf_source_rf = RandomForestClassifier(n_estimators=RF_SIZE)
    clf_source_dt.fit(Xs, ys)
    clf_source_rf.fit(Xs, ys)
    #score_src_src = clf_source.score(Xs, ys)
    #score_src_trgt = clf_source.score(Xt_test, yt_test)
    #print('Training score Source model: {:.3f}'.format(score_src_src))
    #print('Testing score Source model: {:.3f}'.format(score_src_trgt))
    clfs = []
    scores = []
    # Transfer with SER
    #clf_transfer = copy.deepcopy(clf_source)
    #transferred_dt = TransferTreeClassifier(estimator=clf_transfer,Xt=Xt,yt=yt)

    for method in methods:
        Nkmin = sum(yt == 0 )
        root_source_values = clf_source_dt.tree_.value[0].reshape(-1)
        props_s = root_source_values
        props_s = props_s / sum(props_s)
        props_t = np.zeros(props_s.size)
        for k in range(props_s.size):
            props_t[k] = np.sum(yt == k) / yt.size

        coeffs = np.divide(props_t, props_s)          

        clf_transfer_dt = copy.deepcopy(clf_source_dt)
        clf_transfer_rf = copy.deepcopy(clf_source_rf)
        
        if method == 'relab':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="")
            transferred_dt.fit(Xt,yt)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="",bootstrap=True)
            transferred_rf.fit(Xt,yt)
        if method == 'ser':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt.set_params(max_depth=10),algo="ser")
            transferred_dt.fit(Xt,yt)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="ser")
            transferred_rf.fit(Xt,yt)
        if method == 'ser_nr':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="ser")
            transferred_dt._ser(Xt, yt,node=0,original_ser=False,no_red_on_cl=True,cl_no_red=[0])
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="ser")
            transferred_rf._ser_rf(Xt, yt,original_ser=False,no_red_on_cl=True,cl_no_red=[0])
        
        # WARNING! Error Raised with this test
        if method == 'ser_no_ext':
            pass
            #decision tree
            #transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="ser")
            #transferred_dt._ser(Xt, yt,node=0,original_ser=False,no_ext_on_cl=True,cl_no_red=[0],ext_cond=True)
            #random forest
            #transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="ser")
            #transferred_rf._ser_rf(Xt, yt,original_ser=False,no_ext_on_cl=True,cl_no_ext=[0],ext_cond=True)
        if method == 'ser_nr_lambda':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="ser")
            transferred_dt._ser(Xt, yt,node=0,original_ser=False,no_red_on_cl=True,cl_no_red=[0],
                                leaf_loss_quantify=True,leaf_loss_threshold=0.5,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="ser")
            transferred_rf._ser_rf(Xt, yt,original_ser=False,no_red_on_cl=True,cl_no_red=[0],
                                leaf_loss_quantify=True,leaf_loss_threshold=0.5,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
        if method == 'strut':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt.fit(Xt,yt)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            transferred_rf.fit(Xt,yt)
        if method == 'strut_nd':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt._strut(Xt, yt,node=0,use_divergence=False)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            transferred_rf._strut_rf(Xt, yt,use_divergence=False)
        if method == 'strut_lambda':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt._strut(Xt, yt,node=0,adapt_prop=True,root_source_values=root_source_values,
                                  Nkmin=Nkmin,coeffs=coeffs)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            transferred_rf._strut_rf(Xt, yt,adapt_prop=True,root_source_values=root_source_values,
                                  Nkmin=Nkmin,coeffs=coeffs)
        if method == 'strut_np':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt._strut(Xt, yt,node=0,adapt_prop=False,no_prune_on_cl=True,cl_no_prune=[0],
                                leaf_loss_quantify=False,leaf_loss_threshold=0.5,no_prune_with_translation=False,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            transferred_rf._strut(Xt, yt,adapt_prop=False,no_prune_on_cl=True,cl_no_prune=[0],
                                leaf_loss_quantify=False,leaf_loss_threshold=0.5,no_prune_with_translation=False,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
        if method == 'strut_lambda_np':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt._strut(Xt, yt,node=0,adapt_prop=False,no_prune_on_cl=True,cl_no_prune=[0],
                                leaf_loss_quantify=False,leaf_loss_threshold=0.5,no_prune_with_translation=False,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
            #random forest
            transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            transferred_rf._strut(Xt, yt,adapt_prop=True,no_prune_on_cl=True,cl_no_prune=[0],
                                leaf_loss_quantify=True,leaf_loss_threshold=0.5,no_prune_with_translation=False,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
        if method == 'strut_lambda_np2':
            #decision tree
            transferred_dt = TransferTreeClassifier(estimator=clf_transfer_dt,algo="strut")
            transferred_dt._strut(Xt, yt,node=0,adapt_prop=False,no_prune_on_cl=True,cl_no_prune=[0],
                                leaf_loss_quantify=False,leaf_loss_threshold=0.5,no_prune_with_translation=False,
                                root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs)
            # Warning! Error Raised because `strut` not in TransferForest
            #random forest
            #transferred_rf = TransferForestClassifier(estimator=clf_transfer_rf,algo="strut")
            #transferred_rf._strut(Xt, yt,adapt_prop=True,no_prune_on_cl=True,cl_no_prune=[0],
            #                    leaf_loss_quantify=True,leaf_loss_threshold=0.5,no_prune_with_translation=True,
            #                    root_source_values=root_source_values,Nkmin=Nkmin,coeffs=coeffs) 
            
        score = transferred_dt.estimator.score(Xt_test, yt_test)
        #score = clf_transfer.score(Xt_test, yt_test)
        print('Testing score transferred model ({}) : {:.3f}'.format(method, score))
        clfs.append(transferred_dt.estimator)
        #clfs.append(clf_transfer)
        scores.append(score)