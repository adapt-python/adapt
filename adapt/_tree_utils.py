import copy
import numpy as np


def _bootstrap_(size,class_wise=False,y=None):   
    if class_wise:
        if y is None:
            print("Error : need labels to apply class wise bootstrap.")
        else:
            inds = []
            oob_inds = []
            classes_ = set(y)
            ind_classes_ = np.zeros(len(classes_),dtype=object)
            
            for j,c in enumerate(classes_):
                ind_classes_[j] = np.where(y==c)[0]
                s = ind_classes_[j].size
                inds += list(np.random.choice(ind_classes_[j], s, replace=True))
                oob_inds += list(set(ind_classes_[j]) - set(inds))
                
            inds,oob_inds = np.array(inds),np.array(oob_inds)
    else:
        inds = np.random.choice(np.arange(size), size, replace=True)
        oob_inds = set(np.arange(size)) - set(inds)
        oob_inds = np.array(list(oob_inds))
    
    return inds, oob_inds
    
def depth_tree(dt,node=0):
    
    if dt.tree_.feature[node] == -2:
        return 0
    else:
        nl = dt.tree_.children_left[node]
        nr = dt.tree_.children_right[node]
        
        return max(depth_tree(dt,nl),depth_tree(dt,nr)) + 1

def depth_rf(rf):
    d = 0
    for p in rf.estimators_:
        d = d + p.tree_.max_depth
    return d/len(rf.estimators_)

def depth(dtree,node):
    p,t,b = extract_rule(dtree,node)
    return len(p)

def depth_array(dtree, inds):
    depths = np.zeros(np.array(inds).size)
    for i, e in enumerate(inds):
        depths[i] = depth(dtree, i)
    return depths

def sub_nodes(tree_, node):
    if (node == -1):
        return list()
    if (tree_.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree_, tree_.children_left[node]) + sub_nodes(tree_, tree_.children_right[node])
    
def find_parent_vtree(tree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:
        
        try:
            p = list(tree.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(tree.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b

def isinrule(rule, split):
    f,t = split
    
    feats, ths, bools = rule
    for k,f2 in enumerate(feats):

        if f2 == f and t == ths[k]:
            return 1,bools[k]
    return 0,0

def isdisj_feat(ths1,bools1,ths2,bools2):
    if np.sum(bools1 == -1) != 0:
        max_th1 = np.amin(ths1[bools1==-1])
    else:
        max_th1 = np.inf
        
    if np.sum(bools1 == 1) != 0:
        min_th1 = np.amax(ths1[bools1==1])
    else:
        min_th1 = - np.inf
    
    if np.sum(bools2 == -1) != 0:
        max_th2 = np.amin(ths2[bools2==-1])
    else: 
        max_th2 = np.inf
        
    if np.sum(bools2 == 1) != 0:
        min_th2 = np.amax(ths2[bools2==1])  
    else:
        min_th2 = - np.inf
    
    if ( min_th2> min_th1 and min_th2< max_th1 ) or ( max_th2> min_th1 and max_th2< max_th1 ) or ( max_th1> min_th2 and max_th1< max_th2 ) or ( min_th1> min_th2 and min_th1< max_th2 ) or ( min_th1 == min_th2 and max_th1 == max_th2 )   :
        return 0
    else:
        return 1
    
def isdisj(rule1,rule2):
    feats1, ths1, bools1 = rule1
    feats2, ths2, bools2 = rule2
    if np.array(rule1).size == 0 or np.array(rule2).size == 0 :
        return 0
    isdj = 0

    for phi in feats1:
        
        if phi in feats2:
            
            ths1_f = ths1[ feats1 == phi ]
            ths2_f = ths2[ feats2 == phi ]
            bools1_f = bools1[ feats1 == phi ]
            bools2_f = bools2[ feats2 == phi ]
            
            if isdisj_feat(ths1_f,bools1_f,ths2_f,bools2_f):
                isdj = 1

    
    return isdj

def extract_rule_vtree(tree,node):
    
    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
            
            feats.append(tree.feature[node])
            ths.append(tree.threshold[node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent_vtree(tree,node)
        
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
    
    return np.array(feats), np.array(ths), np.array(bools)

def extract_rule(dtree,node):
    
    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
            
            feats.append(dtree.tree_.feature[node])
            ths.append(dtree.tree_.threshold[node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent(dtree,node)
        
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
    
    return np.array(feats), np.array(ths), np.array(bools)


def extract_leaves_rules(dtree):
    leaves = np.where(dtree.tree_.feature == -2)[0]
    
    rules = np.zeros(leaves.size,dtype = object)
    for k,f in enumerate(leaves) :
        rules[k] = extract_rule(dtree,f)
    
    return leaves, rules


def find_parent(dtree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:
        
        try:
            p = list(dtree.tree_.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(dtree.tree_.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b

def add_to_parents(dTree, node, values):
    p, b = find_parent(dTree.tree_, node)
    if b != 0:
        dTree.tree_.value[p] = dTree.tree_.value[p] + values
        add_to_parents(dTree, p, values)
        
def leaf_error(tree, node):
    if np.sum(tree.value[node]) == 0:
        return 0
    else:
        return 1 - np.max(tree.value[node]) / np.sum(tree.value[node])


def error(tree, node):
    if node == -1:
        return 0
    else:
        
        if tree.feature[node] == -2:
            return leaf_error(tree, node)
        else:
            # Pas une feuille
            
            nr = np.sum(tree.value[tree.children_right[node]])
            nl = np.sum(tree.value[tree.children_left[node]])
            
            if nr + nl == 0:
                return 0
            else:
                er = error(tree, tree.children_right[node])
                el = error(tree, tree.children_left[node])
                
                return (el * nl + er * nr) / (nl + nr)

def KL_divergence(class_counts_P,
                  class_counts_Q):
    # KL Divergence to assess the difference between two distributions
    # Definition: $D_{KL}(P||Q) = \sum{i} P(i)ln(\frac{P(i)}{Q(i)})$
    # epsilon to avoid division by 0
    epsilon = 1e-8
    class_counts_P += epsilon
    class_counts_Q += epsilon
    P = class_counts_P * 1. / class_counts_P.sum()
    Q = class_counts_Q * 1. / class_counts_Q.sum()
    Dkl = (P * np.log(P * 1. / Q)).sum()
    return Dkl

def H(class_counts):
    # Entropy
    # Definition: $H(P) = \sum{i} -P(i) ln(P(i))$
    epsilon = 1e-8
    class_counts += epsilon
    P = class_counts * 1. / class_counts.sum()
    return - (P * np.log(P)).sum()


def IG(class_counts_parent,
       class_counts_children):
    # Information Gain
    H_parent = H(class_counts_parent)
    H_children = np.asarray([H(class_counts_child) for class_counts_child in class_counts_children])
    N = class_counts_parent.sum()
    p_children = np.asarray([class_counts_child.sum() * 1. / N for class_counts_child in class_counts_children])
    information_gain = H_parent - (p_children * H_children).sum()
    return information_gain


def JSD(P, Q):
    M = (P + Q) * 1. / 2
    Dkl_PM = KL_divergence(P, M)
    Dkl_QM = KL_divergence(Q, M)
    return (Dkl_PM + Dkl_QM) * 1. / 2


def DG(Q_source_l,
       Q_source_r,
       Q_target_l,
       Q_target_r):
    # compute proportion of instances at left and right
    p_l = Q_target_l.sum()
    p_r = Q_target_r.sum()
    total_counts = p_l + p_r
    p_l /= total_counts
    p_r /= total_counts
    # compute the DG
    return 1. - p_l * JSD(Q_target_l, Q_source_l) - p_r * JSD(Q_target_r, Q_source_r)



def GINI(class_distribution):
    if class_distribution.sum():
        p = class_distribution / class_distribution.sum()
        return 1 - (p**2).sum()
    return 0


def threshold_selection(Q_source_parent,
                        Q_source_l,
                        Q_source_r,
                        X_target_node,
                        Y_target_node,
                        phi,
                        classes,
                        use_divergence=True,
                        measure_default_IG=True):
    # print("Q_source_parent : ", Q_source_parent)
    # sort the corrdinates of X along phi
    #X_phi_sorted = np.sort(X_target_node[:, phi])
    
    #Consider dinstinct values for tested thresholds
    X_phi_sorted = np.array(list(set(X_target_node[:, phi])))
    X_phi_sorted = np.sort(X_phi_sorted)
    
    nb_tested_thresholds = X_phi_sorted.shape[0] - 1
    
    if nb_tested_thresholds == 0:
        return X_phi_sorted[0]
    
    measures_IG = np.zeros(nb_tested_thresholds)
    measures_DG = np.zeros(nb_tested_thresholds)
    for i in range(nb_tested_thresholds):
        threshold = (X_phi_sorted[i] + X_phi_sorted[i + 1]) * 1. / 2
        Q_target_l, Q_target_r = compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           threshold,
                                                           classes)

        measures_IG[i] = IG(Q_source_parent,
                            [Q_target_l, Q_target_r])
        measures_DG[i] = DG(Q_source_l,
                            Q_source_r,
                            Q_target_l,
                            Q_target_r)
    index = 0
    max_found = 0
    
    if use_divergence:
        for i in range(1, nb_tested_thresholds - 1):
            if measures_IG[i] >= measures_IG[i - 1] and measures_IG[i] >= measures_IG[i + 1] and measures_DG[i] > measures_DG[index]:
                max_found = 1
                index = i
                
        if not max_found :

            if measure_default_IG:
                index = np.argmax(measures_IG)
            else:
                index = np.argmax(measures_DG)
    else:    
        index = np.argmax(measures_IG)
    

    threshold = (X_phi_sorted[index] + X_phi_sorted[index + 1]) * 1. / 2
    return threshold
# =============================================================================
# 
# =============================================================================
    
def get_children_distributions(decisiontree,
                               node_index):
    tree_ = decisiontree.tree_
    child_l = tree_.children_left[node_index]
    child_r = tree_.children_right[node_index]
    Q_source_l = tree_.value[child_l]
    Q_source_r = tree_.value[child_r]
    return [np.asarray(Q_source_l), np.asarray(Q_source_r)]


def get_node_distribution(decisiontree,
                          node_index):
    tree_ = decisiontree.tree_
    Q = tree_.value[node_index]
    return np.asarray(Q)


def compute_class_distribution(classes,
                               class_membership):
    unique, counts = np.unique(class_membership,
                               return_counts=True)
    classes_counts = dict(zip(unique, counts))
    classes_index = dict(zip(classes, range(len(classes))))
    distribution = np.zeros(len(classes))
    for label, count in classes_counts.items():
        class_index = classes_index[label]
        distribution[class_index] = count
    return distribution


def compute_Q_children_target(X_target_node,
                              Y_target_node,
                              phi,
                              threshold,
                              classes):
    # Split parent node target sample using the threshold provided
    # instances <= threshold go to the left
    # instances > threshold go to the right
    decision_l = X_target_node[:, phi] <= threshold
    decision_r = np.logical_not(decision_l)
    Y_target_child_l = Y_target_node[decision_l]
    Y_target_child_r = Y_target_node[decision_r]
    Q_target_l = compute_class_distribution(classes, Y_target_child_l)
    Q_target_r = compute_class_distribution(classes, Y_target_child_r)
    return Q_target_l, Q_target_r


# =============================================================================
# 
# =============================================================================

def compute_LLR_estimates_homog(decisiontree,cl=1,node=0,Nkmin=1,root_source_values=None):

    if root_source_values is None:
         root_source_values = decisiontree.tree_.value[0,0,:]
        
    if decisiontree.tree_.feature[node]== -2 :   
        ps = decisiontree.tree_.value[node,0,:]/sum(decisiontree.tree_.value[node,0,:])            
        p1_in_l = decisiontree.tree_.value[node,0,cl]/root_source_values[cl]        
    
        return [np.power(1 - p1_in_l,Nkmin)], [ps]
    else:
        child_l = decisiontree.tree_.children_left[node]
        child_r = decisiontree.tree_.children_right[node]
        
        comp_l,p_l = compute_LLR_estimates_homog(decisiontree,cl=cl,node=child_l,Nkmin=Nkmin,root_source_values=root_source_values) 
        comp_r,p_r = compute_LLR_estimates_homog(decisiontree,cl=cl,node=child_r,Nkmin=Nkmin,root_source_values=root_source_values) 
        
        return comp_l + comp_r, p_l + p_r
    
def contain_leaf_to_not_prune(decisiontree,cl=1,node=0,Nkmin=1,threshold=1,coeffs=None,root_source_values=None):
    
    risks,source_probs = compute_LLR_estimates_homog(decisiontree,cl=cl,node=node,Nkmin=Nkmin,root_source_values=root_source_values)

    bools_thresh = np.array(risks) > threshold 
    bools_maj_cl = np.zeros(bools_thresh.size)
    
    for k,ps in enumerate(source_probs):
        bools_maj_cl[k] = np.argmax(np.multiply(coeffs,ps)) == cl 

    if sum(bools_maj_cl*bools_thresh.reshape(-1)) > 0 :
        return True
    else:
        return False
    
# =============================================================================
# 
# =============================================================================    
    

def coherent_new_split(phi,th,rule):
    #coherent_regardless_class = 0
    #still_splitting = 1
    inrule, sense = isinrule(rule,(phi,th))

    if inrule:
        return 0,sense
    
    feats, ths, bools = rule
    
    if phi not in feats:
        return 1,0
    else:
        if np.sum((feats == phi)*(bools==-1)) != 0:
            max_th = np.amin(ths[(feats == phi)*(bools==-1)])
        else:
            max_th = np.inf
        
        if np.sum((feats == phi)*(bools==1)) != 0:
            min_th = np.amax(ths[(feats == phi)*(bools==1)])
        else:
            min_th = - np.inf
        
        if th >= max_th :
            return 0,-1
        elif th <= min_th:
            return 0,1
        else:
            return 1,0
      
def all_coherent_splits(rule,all_splits):

    inds = np.zeros(all_splits.shape[0],dtype=bool)
    splits = copy.copy(all_splits)
    for c,split in enumerate(all_splits):
        phi, th = split
        #if not isinrule(rule,split):
        coh,sense = coherent_new_split(phi,th,rule)
        if coh:
            inds[c] = 1
       
    return splits[inds]

def new_random_split(p,all_splits):
    inds = np.arange(0,all_splits.shape[0])
    ind = int(np.random.choice(inds,p=p))
    return all_splits[ind]


def filter_feature(splits,feats):
    positive_splits = list()
    negative_splits = list()
    
    for s in splits :
        phi,th = s
        if phi in feats:
            positive_splits.append(s)
        else:
            negative_splits.append(s)
            
    return np.array(positive_splits), np.array(negative_splits)

# =============================================================================
# 
# =============================================================================
def liste_non_coherent_splits(dtree,rule,node=0):
    
    indexes_subtree = sub_nodes(dtree.tree_,node)
    
    phis = list()
    ths = list()
    b = list()
    indexes = list()
    
    for n in indexes_subtree:
        phi,th = dtree.tree_.feature[n],dtree.tree_.threshold[n]
        coh,non_coherent_sense = coherent_new_split(phi,th,rule)
        if not coh :
            phis.append(phi)
            ths.append(th)
            b.append(non_coherent_sense)
            indexes.append(n)
            
    return indexes,phis,ths,b

def bounds_rule(rule,n_features):
    phis,ths,bs = rule
    bound_infs = np.repeat(-np.inf,n_features)
    bound_sups = np.repeat(np.inf,n_features)
    
    for k,p in enumerate(phis) :
        if bs[k] == -1 and bound_sups[int(p)] > ths[k]:
            bound_sups[int(p)]=ths[k]
        if bs[k] == 1 and bound_infs[int(p)] < ths[k]:
            bound_infs[int(p)]=ths[k]
            
    return bound_infs,bound_sups

