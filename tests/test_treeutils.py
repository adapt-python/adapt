import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import adapt._tree_utils as ut

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
classes_test = [0,1]
node_test = 5
node_test2 = 4
feats_test = np.array([0,1])
values_test = np.array([5,10])
clf_source_dt = DecisionTreeClassifier(max_depth=None)
clf_source_rf = RandomForestClassifier(n_estimators=RF_SIZE)
clf_source_dt.fit(Xs, ys)
clf_source_rf.fit(Xs, ys)

def test_depth():
    ut.depth_tree(clf_source_dt)
    ut.depth_rf(clf_source_rf)
    ut.depth(clf_source_dt,node_test)
    ut.depth_array(clf_source_dt,np.arange(clf_source_dt.tree_.node_counte_count))

def test_rules():
    
    ut.sub_nodes(clf_source_dt.tree_,node_test)
    parent,direction = ut.find_parent_vtree(clf_source_dt.tree_, node_test)
    parent,direction = ut.find_parent(clf_source_dt, node_test)
    p,t,b = ut.extract_rule_vtree(clf_source_dt.tree_,node_test)
    p,t,b = ut.extract_rule(clf_source_dt,node_test)
    p2,t2,b2 = ut.extract_rule(clf_source_dt,node_test2)
    
    rule = p,t,b
    rule2 = p2,t2,b2
    split_0 = p[0],t[0]

    ut.isinrule(rule, split_0)
    ut.isdisj_feat(p[0],t[0],p[1],t[1])
    ut.isdisj(rule,rule2)

    ut.bounds_rule(rule,clf_source_dt.n_features_)
    
    leaves,rules = ut.extract_leaves_rules(clf_source_dt)
    ut.add_to_parents(clf_source_dt, node_test, values_test)
    
def test_splits():
    leaves,rules = ut.extract_leaves_rules(clf_source_dt)
    p,t,b = ut.extract_rule(clf_source_dt,node_test)
    p2,t2,b2 = ut.extract_rule(clf_source_dt,node_test2)
    rule = p,t,b
    rule2 = p2,t2,b2
    
    ut.coherent_new_split(p[1],t[1],rule2)
    ut.liste_non_coherent_splits(clf_source_dt,rule)
    
    all_splits = np.zeros(clf_source_dt.tree_.node_count - leaves.size,dtype=[("phi",'<i8'),("th",'<f8')])
    coh_splits = ut.all_coherent_splits(rule,all_splits)
    s = coh_splits.size
    
    ut.filter_feature(all_splits,feats_test)
    ut.new_random_split(np.ones(s)/s,coh_splits)
    
def test_error():
    e = ut.error(clf_source_dt,node_test)
    le = ut.leaf_error(clf_source_dt,node_test)
    return e,le

def test_distribution():
    ut.get_children_distributions(clf_source_dt,node_test)
    ut.get_node_distribution(clf_source_dt,node_test)
    ut.compute_class_distribution(classes_test,ys)
    
    phi = clf_source_dt.tree_.feature[0]
    threshold = clf_source_dt.tree_.threshold[0]
    ut.compute_Q_children_target(Xs,ys,phi,threshold,classes_test)

def test_pruning_risk():
    ut.compute_LLR_estimates_homog(clf_source_dt)
    ut.contain_leaf_to_not_prune(clf_source_dt)
    
def test_divergence_computation():
    phi = clf_source_dt.tree_.feature[0]
    threshold = clf_source_dt.tree_.threshold[0]

    Q_source_parent = ut.get_node_distribution(clf_source_dt,node_test)     
    Q_source_l, Q_source_r = ut.get_children_distributions(clf_source_dt,node_test)
    Q_target_l, Q_target_r = ut.compute_Q_children_target(Xt,yt,phi,threshold,classes_test)

   
    ut.H(Q_source_parent)
    ut.GINI(Q_source_parent)
    ut.IG(Q_source_parent,[Q_target_l, Q_target_r])    
    ut.DG(Q_source_l,Q_source_r,Q_target_l,Q_target_r)
    ut.JSD(Q_target_l, Q_source_l)
    
    ut.KL_divergence(Q_source_l,Q_target_l)
    ut.threshold_selection(Q_source_parent,Q_source_l,Q_source_r,Xt,yt,phi,classes_test)


    