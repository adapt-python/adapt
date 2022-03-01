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
    H_children = np.asarray([H(class_counts_child)
                             for class_counts_child in class_counts_children])
                             N = class_counts_parent.sum()
                             p_children = np.asarray([class_counts_child.sum(
                                                      ) * 1. / N for class_counts_child in class_counts_children])
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


