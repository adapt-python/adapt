#from adapt.utils import (check_arrays,set_random_seed,check_estimator)
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import tree_utils as ut

class TransferTreeClassifier:
    """
    TransferTreeClassifier
    

    Parameters
    ----------    
    estimator : sklearn DecsionTreeClassifier (default=None)
        Source decision tree classifier.
        
    Xt : numpy array (default=None)
        Target input data.
        
    yt : numpy array (default=None)
        Target output data.
        
    algo : str or callable (default="")
        Leaves relabeling if "" or "relab". 
        "ser" and "strut" for SER and STRUT algorithms
        
    (pas la peine de commenter Xt, yt, copy, verbose et random_state)
        
    Attributes
    ----------
    estimator : sklearn DecsionTreeClassifier
        Transferred decision tree classifier using target data.
        
    source_model:
        Source decision tree classifier.
        
    parents : numpy array of int.
        
    bool_parents_lr : numpy array of {-1,0,1} values.
        
    paths : numpy array of int arrays.
        
    rules : numpy array of 3-tuple arrays.
        
    depths : numpy array of int.
        
        
    Examples
    --------
    >>> import numpy as np
    >>> from adapt.parameter_based import TransferTreeClassifier
    >>> from sklearn.tree import DecsionTreeClassifier
    >>> np.random.seed(0)
    >>> Xs = np.random.randn(50) * 0.1
    >>> Xs = np.concatenate((Xs, Xs + 1.))
    >>> Xt = np.random.randn(100) * 0.1
    >>> ys = (Xs < 0.1).astype(int)
    >>> yt = (Xt < 0.05).astype(int)
    >>> lc = DecsionTreeClassifier()
    >>> lc.fit(Xs.reshape(-1, 1), ys)
    >>> lc.score(Xt.reshape(-1, 1), yt)
    0.67
    >>> rt = TransferTreeClassifier(lc, random_state=0)
    >>> rt.fit(Xt[:10].reshape(-1, 1), yt[:10].reshape(-1, 1))
    >>> rt.estimator_.score(Xt.reshape(-1, 1), yt)
    0.67

    References
    ----------
    .. [1] `[1] <https://ieeexplore.ieee.org/document/7592407>`_ Segev, Noam and Harel, Maayan \
Mannor, Shie and Crammer, Koby and El-Yaniv, Ran \
"Learn on Source, Refine on Target: A Model Transfer Learning Framework with Random Forests". In IEEE TPAMI, 2017.
    .. [2] `[2] <https://ieeexplore.ieee.org/document/8995296>`_ Minvielle, Ludovic and Atiq, Mounir \
Peignier, Sergio and Mougeot, Mathilde \
"Transfer Learning on Decision Tree with Class Imbalance". In IEEE ICTAI, 2019.
    """
    
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 algo="",
                 cpy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
#        if not hasattr(estimator, "tree_"):
#            raise NotFittedError("`estimator` argument has no ``tree_`` attribute, "
#                                 "please call `fit` on `estimator` or use "
#                                 "another estimator.")
        
        self.parents = np.zeros(estimator.tree_.node_count,dtype=int)
        self.bool_parents_lr = np.zeros(estimator.tree_.node_count,dtype=int)
        self.rules = np.zeros(estimator.tree_.node_count,dtype=object)
        self.paths = np.zeros(estimator.tree_.node_count,dtype=object)
        self.depths = np.zeros(estimator.tree_.node_count,dtype=int)
        
        self.estimator = estimator
        self.source_model = copy.deepcopy(self.estimator)
        
        self.Xt = Xt
        self.yt = yt
        self.algo = algo
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        self.params = params

        #Init. meta params
        self._compute_params()
  
        #Target model
        if Xt is not None and yt is not None:
            self._relab(Xt,yt)
            self.target_model = self.estimator
            self.estimator = copy.deepcopy(self.source_model)
        else:
            self.target_model = None
        
    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit TransferTreeClassifier.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Arguments for the estimator.

        Returns
        -------
        self : returns an instance of self
        """
        
        #if self.estimator is None:
        #Pas d'arbre source
        
        #if self.estimator.node_count == 0:
        #Arbre vide
        
        #set_random_seed(self.random_state)
        #Xt, yt = check_arrays(Xt, yt)
        
        #self.estimator_ = check_estimator(self.estimator,copy=self.copy,force_copy=True)
        
        #Tree_ = self.estimator.tree_
        
        #Target model :
        if self.target_model is None :
            if Xt is not None and yt is not None:
                self._relab(Xt,yt)
                self.target_model = self.estimator
                self.estimator = copy.deepcopy(self.source_model)

        self._modify_tree(self.estimator, Xt, yt)
        
        return self

    
    def _modify_tree(self, dtree, X, y):
        
        # Aiguillage
        if self.algo == "" or self.algo == "relabel":
            return self._relab(X, y)
        elif self.algo == "ser":
            return self._ser(X, y)
        
        elif self.algo == "strut":
            return self._strut(X, y)
        
        elif hasattr(self.algo, "__call__"):
            return self.algo(dtree, X, y)

    ### @@@ ###

    ###########
    
    def _compute_params(self,node=0):
        #Tree_ = self.estimator.tree_
      
        if node == 0 :
            #default values
            self.parents[0] = -1
            self.rules[0] = (np.array([]),np.array([]),np.array([]))
            self.paths[0] = np.array([])
        else:
            parent,b = ut.find_parent(self.estimator, node)
            self.parents[node] = parent
            self.bool_parents_lr[node] = b
            self.depths[node] = self.depths[parent]+1
            
            self.paths[node] = np.array(list(self.paths[parent])+[parent])
            
            (features,thresholds,bs) = self.rules[parent]
            new_f=np.zeros(features.size+1,dtype=int)
            new_t=np.zeros(thresholds.size+1,dtype=float)
            new_b=np.zeros(bs.size+1,dtype=int)
            new_f[:-1] = features
            new_t[:-1] = thresholds
            new_b[:-1] = bs
            new_f[-1] = self.estimator.tree_.feature[parent]
            new_t[-1] = self.estimator.tree_.threshold[parent]
            new_b[-1] = b
            self.rules[node] = (new_f,new_t,new_b)

        if self.estimator.tree_.feature[node] != -2:
            child_l = self.estimator.tree_.children_left[node]
            child_r = self.estimator.tree_.children_right[node]
            dl = self._compute_params(node=child_l)
            dr = self._compute_params(node=child_r)
            return max(dl,dr)
        else:
            return self.depths[node]
            

                
    def _update_split(self,node,feature,threshold):
  
        #Tree_ = self.estimator.tree_
        self.estimator.tree_.feature[node] = feature
        self.estimator.tree_.threshold[node] = threshold

        for k in ut.sub_nodes(self.estimator.tree_, node)[1:]:
            ind_ = list(self.paths[k]).index(node)
            (p,t,b) = self.rules[k]
            (p[ind_],t[ind_]) =  (feature,threshold)
            self.rules[k] = (p,t,b)
             
        return node
    
    def _cut_leaf(self,node,leaf_value=None):

        #dTree = self.estimator
        dic = self.estimator.tree_.__getstate__().copy()
        dic_old = dic.copy()
        size_init = self.estimator.tree_.node_count

        node_to_rem = ut.sub_nodes(self.estimator.tree_, node)[1:]

        inds = list(set(np.arange(size_init)) - set(node_to_rem))
        
        dic['capacity'] = self.estimator.tree_.capacity - len(node_to_rem)
        dic['node_count'] = self.estimator.tree_.node_count - len(node_to_rem)
        
        dic['nodes']['feature'][node] = -2
        dic['nodes']['left_child'][node] = -1
        dic['nodes']['right_child'][node] = -1
            
        
        left_old = dic_old['nodes']['left_child']
        right_old = dic_old['nodes']['right_child']
        dic['nodes'] = dic['nodes'][inds]
        dic['values'] = dic['values'][inds]

        old_parents = self.parents.copy()
        old_paths = self.paths.copy()
        
        self.parents = self.parents[inds]
        self.bool_parents_lr = self.bool_parents_lr[inds]
        self.rules = self.rules[inds]
        self.paths = self.paths[inds]
        self.depths = self.depths[inds]
        
        max_d = np.max(self.depths)
        dic['max_depth'] = max_d
        
        if leaf_value is not None:
            dic['values'][node] = leaf_value
        
        for i, new in enumerate(inds):
            if new != 0 and i!=0 :
                self.parents[i] = inds.index(old_parents[new])
                for z,u in enumerate(self.paths[i]):
                    self.paths[i][z] = inds.index(old_paths[new][z])
            if (left_old[new] != -1):
                dic['nodes']['left_child'][i] = inds.index(left_old[new])
            else:
                dic['nodes']['left_child'][i] = -1
            if (right_old[new] != -1):
                dic['nodes']['right_child'][i] = inds.index(right_old[new])
            else:
                dic['nodes']['right_child'][i] = -1

        (Tree, (n_f, n_c, n_o), b) = self.estimator.tree_.__reduce__()
        del dic_old
        del self.estimator.tree_

        self.estimator.tree_ = Tree(n_f, n_c, n_o)
        self.estimator.tree_.__setstate__(dic)
        
        self.estimator.tree_.max_depth = max_d
        return inds.index(node)
    
    def _cut_left_right(self,node,lr):   
        
        #dTree = self.estimator
        if lr == 1:
            cut_leaf = self._cut_leaf(self.estimator.tree_.children_right[node])
            node = self.parents[cut_leaf]
            repl_node = self.estimator.tree_.children_left[node]
            
        elif lr == -1:
            cut_leaf = self._cut_leaf(self.estimator.tree_.children_left[node])
            node = self.parents[cut_leaf]
            repl_node = self.estimator.tree_.children_right[node]
        
        dic = self.estimator.tree_.__getstate__().copy()
        size_init = self.estimator.tree_.node_count
        node_to_rem = [node,cut_leaf]
        inds = list(set(np.arange(size_init)) - set(node_to_rem))
        
        p, b = self.parents[node],self.bool_parents_lr[node]
  
        dic['capacity'] = self.estimator.tree_.capacity - len(node_to_rem)
        dic['node_count'] = self.estimator.tree_.node_count - len(node_to_rem)

        if p != -1 :
            if b == 1:
                dic['nodes']['right_child'][p] = repl_node
            elif b == -1:
                dic['nodes']['left_child'][p] = repl_node
            else:
                print('Error : need node direction with regard to its parent.')
            
        self.parents[repl_node] = p
        self.bool_parents_lr[repl_node] = b
        
        for k in ut.sub_nodes(self.estimator.tree_, repl_node):
            ind_ = list(self.paths[k]).index(node)
            self.paths[k] = np.delete(self.paths[k],ind_) 
            (f,t,b) = self.rules[k]
            self.rules[k] = (np.delete(f,ind_),np.delete(t,ind_),np.delete(b,ind_))
            self.depths[k] = self.depths[k] - 1
            
        dic_old = dic.copy()
        left_old = dic_old['nodes']['left_child']
        right_old = dic_old['nodes']['right_child']
        dic['nodes'] = dic['nodes'][inds]
        dic['values'] = dic['values'][inds]
        
        old_parents = self.parents.copy()
        old_paths = self.paths.copy()
        
        self.parents = self.parents[inds]
        self.bool_parents_lr = self.bool_parents_lr[inds]
        self.rules = self.rules[inds]
        self.paths = self.paths[inds]
        self.depths = self.depths[inds]
        
        max_d = np.max(self.depths)
        dic['max_depth'] = max_d
        
        for i, new in enumerate(inds):
            if new != 0 and i!=0:
                self.parents[i] = inds.index(old_parents[new])
                for z,u in enumerate(self.paths[i]):
                    self.paths[i][z] = inds.index(old_paths[new][z])
            if (left_old[new] != -1):
                dic['nodes']['left_child'][i] = inds.index(left_old[new])
            else:
                dic['nodes']['left_child'][i] = -1
            if (right_old[new] != -1):
                dic['nodes']['right_child'][i] = inds.index(right_old[new])
            else:
                dic['nodes']['right_child'][i] = -1

        (Tree, (n_f, n_c, n_o), b) = self.estimator.tree_.__reduce__()
        del self.estimator.tree_
        del dic_old
    
        self.estimator.tree_ = Tree(n_f, n_c, n_o)
        self.estimator.tree_.__setstate__(dic)

        self.estimator.tree_.max_depth = max_d
        
        return inds.index(repl_node)

    def _extend(self,node,subtree):
        """adding tree tree2 to leaf f of tree tree1"""
        
        #tree1 = self.estimator.tree_
        tree2 = subtree.tree_
        size_init = self.estimator.tree_.node_count
        
        dic = self.estimator.tree_.__getstate__().copy()
        dic2 = tree2.__getstate__().copy()
        size2 = tree2.node_count
        
        size_init = self.estimator.tree_.node_count

        if self.depths[node] + dic2['max_depth'] > dic['max_depth']:
            dic['max_depth'] = self.depths[node] + tree2.max_depth
        
        dic['capacity'] = self.estimator.tree_.capacity + tree2.capacity - 1
        dic['node_count'] = self.estimator.tree_.node_count + tree2.node_count - 1
        
        dic['nodes'][node] = dic2['nodes'][0]
        
        if (dic2['nodes']['left_child'][0] != - 1):
            dic['nodes']['left_child'][node] = dic2['nodes']['left_child'][0] + size_init - 1
        else:
            dic['nodes']['left_child'][node] = -1
        if (dic2['nodes']['right_child'][0] != - 1):
            dic['nodes']['right_child'][node] = dic2['nodes']['right_child'][0] + size_init - 1
        else:
            dic['nodes']['right_child'][node] = -1
    
        # Attention vecteur impurity pas mis à jour
    
        dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
        dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][size_init:] != -1) * (dic['nodes']['left_child'][size_init:] + size_init) - 1
        dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][size_init:] != -1) * (dic['nodes']['right_child'][size_init:] + size_init) - 1
        
        values = np.concatenate((dic['values'], np.zeros((dic2['values'].shape[0] - 1, dic['values'].shape[1], dic['values'].shape[2]))), axis=0)

        dic['values'] = values

        (Tree, (n_f, n_c, n_o), b) = self.estimator.tree_.__reduce__()

        self.estimator.tree_ = Tree(n_f, n_c, n_o)
        self.estimator.tree_.__setstate__(dic)
        del dic2
        del tree2

        try:
            self.estimator.tree_.value[size_init:, :, subtree.classes_.astype(int)] = subtree.tree_.value[1:, :, :]
        except IndexError as e:
            print("IndexError : size init : ", size_init,
                  "\ndTree2.classes_ : ", subtree.classes_)
            print(e)
        
        self.parents = np.concatenate((self.parents, np.zeros(size2 - 1,dtype=int) ))
        self.bool_parents_lr = np.concatenate((self.bool_parents_lr, np.zeros(size2 - 1,dtype=int) ))
        self.rules = np.concatenate((self.rules, np.zeros(size2 - 1,dtype=object) ))
        self.paths = np.concatenate((self.paths, np.zeros(size2 - 1,dtype=object) ))
        self.depths = np.concatenate((self.depths, np.zeros(size2 - 1,dtype=int) ))
        
        self._compute_params(node=node)
        self.estimator.max_depth = self.estimator.tree_.max_depth

        return node

    def _force_coherence(self,rule,node=0,Translate=False,indexes_nodes=list(),drifts=list(),auto_drift=True):
              
        #dtree = self.estimator
        D_MARGIN = 1
        if Translate and not auto_drift:
            if len(indexes_nodes) != len(drifts):
                print('Error in parameter size for drifts')
                return node
            else:
                for k,n in enumerate(indexes_nodes):
                    self.updateSplit(n,self.estimator.tree_.feature[n],self.estimator.tree_.threshold[n]+drifts[k])
        
        phis,ths,bs = rule
        non_coherent_sense = 0
        
        phi,th = self.estimator.tree_.feature[node],self.estimator.tree_.threshold[node]
        
        if phi != -2:
        #if it is not a leaf
            coh,non_coherent_sense = ut.coherent_new_split(phi,th,rule)
                        
            if not coh:
                if Translate :
                    if auto_drift:
                        b_infs,b_sups = ut.bounds_rule(rule,self.estimator.n_features_)
                        if non_coherent_sense == -1:
                            if b_sups[phi] == np.inf:
                                self.updateSplit(node,phi,th+D_MARGIN)
                            else:
                                self.updateSplit(node,phi,( b_infs[phi] + b_sups[phi] )/2)
                        if non_coherent_sense == 1:
                            if b_infs[phi] == -np.inf:
                                self.updateSplit(node,phi,th-D_MARGIN)
                            else:
                                self.updateSplit(node,phi,( b_infs[phi] + b_sups[phi] )/2)
                    else:                
                        print('Warning:this translation made incoherent subtree')
                
                else:
                    while not coh:
                        node = self.prune(node,include_node=True,lr=non_coherent_sense)
                        phi,th = self.estimator.tree_.feature[node],self.estimator.tree_.threshold[node]
                        coh,non_coherent_sense = ut.coherent_new_split(phi,th,rule)
    
            node_l = self.estimator.tree_.children_left[node]   
            rule_l = self.rules[node_l]
            if self.estimator.tree_.feature[node_l] != -2 :
                node_l = self._force_coherence(rule_l,node=node_l,Translate=Translate,
                                             indexes_nodes=indexes_nodes,drifts=drifts,auto_drift=auto_drift)

            node = self.parents[node_l]

            node_r = self.estimator.tree_.children_right[node]  
            rule_r = self.rules[node_r]
            if self.estimator.tree_.feature[node_r] != -2 :
                node_r = self._force_coherence(rule_r,node=node_r,Translate=Translate,
                                             indexes_nodes=indexes_nodes,drifts=drifts,auto_drift=auto_drift)
            node = self.parents[node_r]  
        
            return node
            
    ### @@@ ###
    
    ###########

    def updateSplit(self,node,feature,threshold):
        return self._update_split(node,feature,threshold)
        
    def updateValue(self,node,values):
        #Tree_ = self.estimator.tree_
        self.estimator.tree_.value[node] = values
        self.estimator.tree_.impurity[node] = ut.GINI(values)
        self.estimator.tree_.n_node_samples[node] = np.sum(values)
        self.estimator.tree_.weighted_n_node_samples[node] = np.sum(values)
        return node
    
    def swap_subtrees(self,node1,node2):
        #Check sub-nodes :
        if node1 == node2:
            print('Warning : same node given twice.')
            return 0
        
        if node2 in ut.sub_nodes(self.estimator.tree_, node1)[1:]:
            print('Error : node2 is a sub-node of node1.')
            return 0

        if node1 in ut.sub_nodes(self.estimator.tree_, node2)[1:]:
            print('Error : node1 is a sub-node of node2.')
            return 0

        p1,b1 = self.parents[node1], self.bool_parents_lr[node1]
        p2,b2 = self.parents[node2], self.bool_parents_lr[node2]
        
        if b1 == -1:
            self.estimator.tree_.children_left[p1] = node2
        elif b1 == 1:
            self.estimator.tree_.children_right[p1] = node2

        if b2 == -1:
            self.estimator.tree_.children_left[p2] = node1
        elif b2 == 1:
            self.estimator.tree_.children_right[p2] = node1
            
        self.parents[node2] = p1
        self.bool_parents_lr[node2] = b1            
        self.parents[node1] = p2
        self.bool_parents_lr[node1] = b2

        d1 = self._compute_params(node=node1) 
        d2 = self._compute_params(node=node2) 
        
        self.estimator.tree_.max_depth = max(d1,d2)
        self.estimator.max_depth = self.estimator.tree_.max_depth
        
        return 1
        
    def prune(self,node,include_node=False,lr=0,leaf_value=None):
        if include_node:
            n = self._cut_left_right(node,lr)
        else:
            n = self._cut_leaf(node,leaf_value=leaf_value)
        return n

    def extend(self,node,subtree):
        n = self._extend(node,subtree)
        return n

    ### @@@ ###

    ###########


    def _relab(self, X_target_node, Y_target_node, node=0):
        
        #Tree_ = self.estimator.tree_
        classes_ = self.estimator.classes_
        
        current_class_distribution = ut.compute_class_distribution(classes_, Y_target_node)
        self.updateValue(node,current_class_distribution)
        
        bool_test = X_target_node[:, self.estimator.tree_.feature[node]] <= self.estimator.tree_.threshold[node]
        not_bool_test = X_target_node[:, self.estimator.tree_.feature[node]] > self.estimator.tree_.threshold[node]

        ind_left = np.where(bool_test)[0]
        ind_right = np.where(not_bool_test)[0]

        X_target_node_left = X_target_node[ind_left]
        Y_target_node_left = Y_target_node[ind_left]

        X_target_node_right = X_target_node[ind_right]
        Y_target_node_right = Y_target_node[ind_right]
        
        if self.estimator.tree_.feature[node] != -2:
            self._relab(X_target_node_left,Y_target_node_left,node=self.estimator.tree_.children_left[node])
            self._relab(X_target_node_right,Y_target_node_right,node=self.estimator.tree_.children_right[node])

        return node


    def _ser(self,X_target_node,y_target_node,node=0,original_ser=True,
             no_red_on_cl=False,cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,ext_cond=None,
             leaf_loss_quantify=False,leaf_loss_threshold=None,coeffs=[1,1],root_source_values=None,Nkmin=None,max_depth=None):
        
        #Tree_ = self.estimator.tree_

        source_values = self.estimator.tree_.value[node].copy()
        #node_source_label = np.argmax(source_values)
        maj_class = np.argmax(self.estimator.tree_.value[node, :].copy())

        if cl_no_red is None:
            old_size_cl_no_red = 0
        else:
            old_size_cl_no_red = np.sum(self.estimator.tree_.value[node][:, cl_no_red])
            
        # Situation où il y a des restrictions sur plusieurs classes ?
        if no_red_on_cl is not None or no_ext_on_cl is not None :
            if no_ext_on_cl:
                cl = cl_no_ext[0]
            if no_red_on_cl:
                cl = cl_no_red[0]

        if leaf_loss_quantify and ((no_red_on_cl  or  no_ext_on_cl) and maj_class == cl) and  self.estimator.tree_.feature[node] == -2 :
            
            ps_rf = self.estimator.tree_.value[node,0,:]/sum(self.estimator.tree_.value[node,0,:])
            p1_in_l = self.estimator.tree_.value[node,0,cl]/root_source_values[cl]
            
            cond_homog_unreached = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
            cond_homog_min_label = np.argmax(np.multiply(coeffs,ps_rf)) == cl
            
        val = np.zeros((self.estimator.n_outputs_, self.estimator.n_classes_))

        for i in range(self.estimator.n_classes_):
            val[:, i] = list(y_target_node).count(i)
        
        self.updateValue(node,val)
        
        if self.estimator.tree_.feature[node]== -2:
            # Extension phase :
            if original_ser:
                if y_target_node.size > 0 and len(set(list(y_target_node))) > 1:
                    
                    if max_depth is not None:
                        d = self.depths[node]
                        DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                        
                    else:
                        DT_to_add = DecisionTreeClassifier()
                        
                    try:
                        DT_to_add.min_impurity_decrease = 0
                    except:
                        DT_to_add.min_impurity_split = 0
                        
                    DT_to_add.fit(X_target_node, y_target_node)
                    self.extend(node, DT_to_add) 
                    
                return node,False
        
            else:
                bool_no_red = False
                cond_extension = False
                    
                if y_target_node.size > 0:
                    
                    if not no_ext_on_cl:
                        if max_depth is not None:
                            d = self.depths[node]
                            DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                        else:
                            DT_to_add = DecisionTreeClassifier()
            
                        try:
                            DT_to_add.min_impurity_decrease = 0
                        except:
                            DT_to_add.min_impurity_split = 0

                        DT_to_add.fit(X_target_node, y_target_node)
                        self.extend(node, DT_to_add) 
                    
                    else:
                        cond_maj = (maj_class not in cl_no_ext)
                        cond_sub_target = ext_cond and (maj_class in y_target_node) and (maj_class in cl_no_ext)
                        cond_leaf_loss = leaf_loss_quantify and maj_class==cl and not (cond_homog_unreached and cond_homog_min_label)
                    
                        cond_extension = cond_maj or cond_sub_target or cond_leaf_loss
                        
                        if cond_extension:
                            if max_depth is not None:
                                d = self.depths[node]
                                DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                            else:
                                DT_to_add = DecisionTreeClassifier()

                            try:
                                DT_to_add.min_impurity_decrease = 0
                            except:
                                DT_to_add.min_impurity_split = 0

                            DT_to_add.fit(X_target_node, y_target_node)
                            self.extend(node, DT_to_add) 
                        
                        else:
                            self.updateValue(node,source_values)
                            
                            ut.add_to_parents(self.estimator, node, source_values) 
                            if no_red_on_cl:
                                bool_no_red = True
                                        
    
                # No red protection with values / used to flag tree parts concerned by pruning restrictions
                if no_red_on_cl and y_target_node.size == 0 and old_size_cl_no_red > 0 and maj_class in cl_no_red:
                    
                    if leaf_loss_quantify :
                        if cond_homog_unreached and cond_homog_min_label :
                            self.updateValue(node,source_values)
                            
                            ut.add_to_parents(self.estimator, node, source_values) 
                            bool_no_red = True
                    else:
                        self.updateValue(node,source_values)

                        ut.add_to_parents(self.estimator, node, source_values) 
                        bool_no_red = True

                return node,bool_no_red
        
        """ From here it cannot be a leaf """
        ### Left / right target computation ###
        bool_test = X_target_node[:, self.estimator.tree_.feature[node]] <= self.estimator.tree_.threshold[node]
        not_bool_test = X_target_node[:, self.estimator.tree_.feature[node]] > self.estimator.tree_.threshold[node]

        ind_left = np.where(bool_test)[0]
        ind_right = np.where(not_bool_test)[0]

        X_target_node_left = X_target_node[ind_left]
        y_target_node_left = y_target_node[ind_left]

        X_target_node_right = X_target_node[ind_right]
        y_target_node_right = y_target_node[ind_right]

        if original_ser:

            new_node_left,bool_no_red_l = self._ser(X_target_node_left,y_target_node_left,node=self.estimator.tree_.children_left[node],original_ser=True,max_depth=max_depth)
            node = self.parents[new_node_left]

            new_node_right,bool_no_red_r = self._ser(X_target_node_right,y_target_node_right,node=self.estimator.tree_.children_right[node],original_ser=True,max_depth=max_depth)
            node = self.parents[new_node_right]
                                  
        else:
            new_node_left,bool_no_red_l = self._ser(X_target_node_left,y_target_node_left,node=self.estimator.tree_.children_left[node],original_ser=False,
                                               no_red_on_cl=no_red_on_cl,cl_no_red=cl_no_red,no_ext_on_cl=no_ext_on_cl,cl_no_ext=cl_no_ext,ext_cond=ext_cond,
                                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,
                                               Nkmin=Nkmin,max_depth=max_depth)


            node = self.parents[new_node_left]

            new_node_right,bool_no_red_r = self._ser(X_target_node_right,y_target_node_right,node=self.estimator.tree_.children_right[node],original_ser=False,
                                               no_red_on_cl=no_red_on_cl,cl_no_red=cl_no_red,no_ext_on_cl=no_ext_on_cl,cl_no_ext=cl_no_ext,ext_cond=ext_cond,
                                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,
                                               Nkmin=Nkmin,max_depth=max_depth)

            node = self.parents[new_node_right]

        if original_ser:
            bool_no_red = False
        else:
            bool_no_red = bool_no_red_l or bool_no_red_r

        le = ut.leaf_error(self.estimator.tree_, node)
        e = ut.error(self.estimator.tree_, node)

        if le <= e:
            if original_ser:
                new_node_leaf = self.prune(node,include_node=False) 
                node = new_node_leaf
            else:
                if no_red_on_cl:
                    if not bool_no_red:
                        new_node_leaf = self.prune(node,include_node=False) 
                        node = new_node_leaf
            
                else:
                    new_node_leaf = self.prune(node,include_node=False) 
                    node = new_node_leaf

        if self.estimator.tree_.feature[node] != -2:

            if original_ser:
                if ind_left.size == 0:
                    node = self.prune(node,include_node=True,lr=-1) 
                    
                if ind_right.size == 0:
                    node = self.prune(node,include_node=True,lr=1)
            else:
                if no_red_on_cl:
                    if ind_left.size == 0 and np.sum(self.estimator.tree_.value[self.estimator.tree_.children_left[node]]) == 0:
                        node = self.prune(node,include_node=True,lr=-1) 
                        
                    if ind_right.size == 0 and np.sum(self.estimator.tree_.value[self.estimator.tree_.children_right[node]]) == 0:
                        node = self.prune(node,include_node=True,lr=1) 
                else:
                    if ind_left.size == 0:
                        node = self.prune(node,include_node=True,lr=-1)
                    
                    if ind_right.size == 0:
                        node = self.prune(node,include_node=True,lr=1) 

        return node,bool_no_red
            

    def _strut(self,X_target_node,Y_target_node,node=0,no_prune_on_cl=False,cl_no_prune=None,adapt_prop=False,
          coeffs=[1, 1],use_divergence=True,measure_default_IG=True,min_drift=None,max_drift=None,no_prune_with_translation=True,
          leaf_loss_quantify=False,leaf_loss_threshold=None,root_source_values=None,Nkmin=None):
                
#        Tree_ = self.estimator.tree_
        
        feature_ = self.estimator.tree_.feature[node]
        classes_ = self.estimator.classes_
        threshold_ = self.estimator.tree_.threshold[node]
            
        old_threshold = threshold_.copy()
        maj_class = np.argmax(self.estimator.tree_.value[node, :].copy())
        
        if min_drift is None or max_drift is None:
            min_drift = np.zeros(self.estimator.n_features_)
            max_drift = np.zeros(self.estimator.n_features_)

        current_class_distribution = ut.compute_class_distribution(classes_, Y_target_node)
        is_reached = (Y_target_node.size > 0)
        no_min_instance_targ = False
        
        if no_prune_on_cl:
            no_min_instance_targ = (sum(current_class_distribution[cl_no_prune]) == 0 )
            is_instance_cl_no_prune = np.sum(self.estimator.tree_.value[node, :,cl_no_prune].astype(int))

        # If it is a leaf :
        if self.estimator.tree_.feature[node] == -2:
            """ When to apply UpdateValue """
            if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) :
                
                ps_rf = self.estimator.tree_.value[node,0,:]/sum(self.estimator.tree_.value[node,0,:])
                p1_in_l = self.estimator.tree_.value[node,0,cl_no_prune]/root_source_values[cl_no_prune]
                cond1 = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
                cond2 = np.argmax(np.multiply(coeffs,ps_rf)) == cl_no_prune
                
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune:
                if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) and not(cond1 and cond2):
                    self.updateValue(node,current_class_distribution)
                    return node
                else:
                    return node
            else:
                self.estimator.tree_.value[node] = current_class_distribution
                return node

        # Only one class remaining in target :
        if (current_class_distribution > 0).sum() == 1:
            """ When to apply Pruning and how if not """
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune :
                bool_subleaf_noprune = True
                if leaf_loss_quantify:
                    bool_subleaf_noprune = ut.contain_leaf_to_not_prune(self.estimator,cl=cl_no_prune,node=node,Nkmin=Nkmin,
                                                                        threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values)
                
                if bool_subleaf_noprune :
                    rule = self.rules[node]
                    if no_prune_with_translation :
                        node = self._force_coherence(rule,node=node,Translate=True,auto_drift=True)
                        return node
                    else:
                        node = self._force_coherence(rule,node=node)
                        return node
                            
                else:
                    node = self.prune(node,include_node=False) 
                    return node

            else:
                node = self.prune(node,include_node=False)
                return node

        # Node unreached by target :
        if not is_reached:
            """ When to apply Pruning and how if not """
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune :
                bool_subleaf_noprune = True
                if leaf_loss_quantify:
                    bool_subleaf_noprune = ut.contain_leaf_to_not_prune(self.estimator,cl=cl_no_prune,node=node,
                                                                     Nkmin=Nkmin,threshold=leaf_loss_threshold,coeffs=coeffs,
                                                                     root_source_values=root_source_values)
                if bool_subleaf_noprune:
                    rule = self.rules[node]
                    
                    if no_prune_with_translation :
                        node = self._force_coherence(rule,node=node,Translate=True,auto_drift=True)
                    else:
                        node = self._force_coherence(rule,node=node)
                else:
                    p,b = self.parents[node], self.bool_parents_lr[node]
                    node = self.prune(p,include_node=True,lr=b) 

            else:
                p,b = self.parents[node], self.bool_parents_lr[node]
                node = self.prune(p,include_node=True,lr=b) 

            return node

        # Node threshold updates :
        """ UpdateValue """
        self.updateValue(node,current_class_distribution)
            
        # update threshold
        if type(threshold_) is np.float64:
            Q_source_l, Q_source_r = ut.get_children_distributions(self.estimator,node)

        Sl = np.sum(Q_source_l)
        Sr = np.sum(Q_source_r)


        if adapt_prop:
            Sl = np.sum(Q_source_l)
            Sr = np.sum(Q_source_r)
            Slt = Y_target_node.size
            Srt = Y_target_node.size
            
            
            D = np.sum(np.multiply(coeffs, Q_source_l))
            Q_source_l = (Slt/Sl)*np.multiply(coeffs,np.divide(Q_source_l,D))
            D = np.sum(np.multiply(coeffs, Q_source_r))
            Q_source_r = (Srt/Sr)*np.multiply(coeffs,np.divide(Q_source_r,D))
    
    
        Q_source_parent = ut.get_node_distribution(self.estimator,node)
            
                                                
        t1 = ut.threshold_selection(Q_source_parent,
                                 Q_source_l.copy(),
                                 Q_source_r.copy(),
                                 X_target_node,
                                 Y_target_node,
                                 feature_,
                                 classes_,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)

        Q_target_l, Q_target_r = ut.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           feature_,
                                                           t1,
                                                           classes_)

        DG_t1 = ut.DG(Q_source_l.copy(),
                   Q_source_r.copy(),
                   Q_target_l,
                   Q_target_r)

        t2 = ut.threshold_selection(Q_source_parent,
                                 Q_source_r.copy(),
                                 Q_source_l.copy(),
                                 X_target_node,
                                 Y_target_node,
                                 feature_,
                                 classes_,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)

        Q_target_l, Q_target_r = ut.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           feature_,
                                                           t2,
                                                           classes_)
            
        DG_t2 = ut.DG(Q_source_r.copy(),Q_source_l.copy(),Q_target_l,Q_target_r)
                                                               
                                                               
        if DG_t1 >= DG_t2:
            self.updateSplit(node,feature_,t1)
        else:
            self.updateSplit(node,feature_,t2)
            # swap children
            child_l = self.estimator.tree_.children_left[node]
            child_r = self.estimator.tree_.children_right[node]
            self.swap_subtrees(child_l,child_r)

        # For No Prune coherence
        ecart = self.estimator.tree_.threshold[node] - old_threshold
        
        if self.estimator.tree_.threshold[node] > old_threshold:
            if ecart > max_drift[self.estimator.tree_.feature[node]] :
                max_drift[self.estimator.tree_.feature[node]] = ecart
        if self.estimator.tree_.threshold[node] < old_threshold:
            if ecart < min_drift[self.estimator.tree_.feature[node]] :
                min_drift[self.estimator.tree_.feature[node]] = ecart

        if self.estimator.tree_.children_left[node] != -1:

            threshold = self.estimator.tree_.threshold[node]
            index_X_child_l = X_target_node[:, feature_] <= threshold
            X_target_child_l = X_target_node[index_X_child_l, :]
            Y_target_child_l = Y_target_node[index_X_child_l]

            node_l = self._strut(X_target_child_l,Y_target_child_l,
                          node=self.estimator.tree_.children_left[node],no_prune_on_cl=no_prune_on_cl,cl_no_prune=cl_no_prune,
                          adapt_prop=adapt_prop,coeffs=coeffs,use_divergence=use_divergence,measure_default_IG=measure_default_IG,
                          min_drift=min_drift.copy(),max_drift=max_drift.copy(),no_prune_with_translation=no_prune_with_translation,
                          leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,root_source_values=root_source_values,Nkmin=Nkmin)
        
            node = self.parents[node_l]

        if self.estimator.tree_.children_right[node] != -1:

            threshold = self.estimator.tree_.threshold[node]
            index_X_child_r = X_target_node[:, feature_] > threshold
            X_target_child_r = X_target_node[index_X_child_r, :]
            Y_target_child_r = Y_target_node[index_X_child_r]
            
            node_r = self._strut(X_target_child_r,Y_target_child_r,
                          node=self.estimator.tree_.children_right[node],no_prune_on_cl=no_prune_on_cl,cl_no_prune=cl_no_prune,
                          adapt_prop=adapt_prop,coeffs=coeffs,use_divergence=use_divergence,measure_default_IG=measure_default_IG,
                          min_drift=min_drift.copy(),max_drift=max_drift.copy(),no_prune_with_translation=no_prune_with_translation,
                          leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,root_source_values=root_source_values,Nkmin=Nkmin)

            node = self.parents[node_r]
              
        return node




class TransferForestClassifier:
    """
    TransferForestClassifier
    
    Parameters
    ----------    
    estimator : sklearn RandomForestClassifier (default=None)
        Source random forest classifier.
        
    Xt : numpy array (default=None)
        Target input data.
        
    yt : numpy array (default=None)
        Target output data.
        
    algo : str or callable (default="")
        Leaves relabeling if "" or "relab". 
        "ser" and "strut" for SER and STRUT algorithms
        
    bootstrap : boolean (default=True).

        
    Attributes
    ----------
    estimator : sklearn RandomForestClassifier
        Transferred random forest classifier using target data.
        
    source_model:
        Source random forest classifier.
        
    target_model:
        Source random forest classifier with relabeled leaves using target data.        
        
    rf_size : int.
        
    estimators_ : numpy array of TransferTreeClassifier.
        
    Examples
    --------
    
    
    References
    ----------
    .. [1] `[1] <https://ieeexplore.ieee.org/document/7592407>`_ Segev, Noam and Harel, Maayan \
Mannor, Shie and Crammer, Koby and El-Yaniv, Ran \
"Learn on Source, Refine on Target: A Model Transfer Learning Framework with Random Forests". In IEEE TPAMI, 2017.
    .. [2] `[2] <https://ieeexplore.ieee.org/document/8995296>`_ Minvielle, Ludovic and Atiq, Mounir \
Peignier, Sergio and Mougeot, Mathilde \
"Transfer Learning on Decision Tree with Class Imbalance". In IEEE ICTAI, 2019.
    """

    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 algo="",
                 bootstrap=False,
                 cpy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
#        if not hasattr(estimator, "tree_"):
#            raise NotFittedError("`estimator` argument has no ``tree_`` attribute, "
#                                 "please call `fit` on `estimator` or use "
#                                 "another estimator.")
        
        self.estimator = estimator
        self.source_model = copy.deepcopy(self.estimator)
        
        self.Xt = Xt
        self.yt = yt
        self.algo = algo
        self.bootstrap = bootstrap
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        self.params = params
        
        self.rf_size = self.estimator.n_estimators
        self.estimators_ = np.zeros(self.rf_size,dtype=object)

        for i in range(self.rf_size):
            self.estimators_[i] = TransferTreeClassifier(estimator = self.estimator.estimators_[i], algo = self.algo)

        #Target model
        if Xt is not None and yt is not None:
            self._relab_rf(Xt,yt)
            self.target_model = self.estimator
            self.estimator = copy.deepcopy(self.source_model)
        else:
            self.target_model = None

    ### @@@ ###

    ###########
    
    def _modify_rf(self, rf, X, y):
        # Aiguillage
        if self.algo == "" or self.algo == "relabel":
            bootstrap = self.bootstrap
            return self._relab_rf(X, y, bootstrap=bootstrap)
        elif self.algo == "ser":
            return self._ser_rf(X, y)        
        elif self.algo == "strut":
            return self._strut_rf(X, y)
        
        elif hasattr(self.algo, "__call__"):
            return self.algo(rf, X, y)

    def fit(self, Xt=None, yt=None, **fit_params):
        """
        Fit TransferTreeClassifier.

        Parameters
        ----------
        Xt : numpy array (default=None)
            Target input data.

        yt : numpy array (default=None)
            Target output data.
            
        fit_params : key, value arguments
            Arguments for the estimator.

        Returns
        -------
        self : returns an instance of self
        """

        #Target model : 
        if self.target_model is None:
            if Xt is not None and yt is not None:
                self._relab_rf(Xt,yt,bootstrap=False)
                self.target_model = self.estimator
                self.estimator = copy.deepcopy(self.source_model)
            
        self._modify_rf(self.estimator, Xt, yt)
        
        return self
    
    ### @@@ ###

    ###########




    ### @@@ ###

    ###########

    def _relab_rf(self, X_target_node, Y_target_node,bootstrap=False):
        
        rf_out = copy.deepcopy(self.source_model)
        
        if bootstrap :             
            inds,oob_inds = ut._bootstrap_(Y_target_node.size,class_wise=True,y=Y_target_node)
            for k in range(self.rf_size):
                X_target_node_bootstrap = X_target_node[inds]
                Y_target_node_bootstrap = Y_target_node[inds]
                self.estimators_[k]._relab(X_target_node_bootstrap, Y_target_node_bootstrap, node=0)
                rf_out.estimators_[k] = self.estimators_[k].estimator
        else:            
            for k in range(self.rf_size):
                self.estimators_[k]._relab(X_target_node, Y_target_node, node=0)
                rf_out.estimators_[k] = self.estimators_[k].estimator
        
        
        self.estimator = rf_out

        return self.estimator
    
    def _ser_rf(self,X_target,y_target,original_ser=True,
             no_red_on_cl=False,cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,ext_cond=None,
             leaf_loss_quantify=False,leaf_loss_threshold=None,coeffs=[1,1],root_source_values=None,Nkmin=None,max_depth=None):
        
        rf_out = copy.deepcopy(self.source_model)
        
        for i in range(self.rf_size):
            root_source_values = None
            coeffs = None
            Nkmin = None
            if  leaf_loss_quantify :    
                Nkmin = sum(y_target == cl_no_red )
                root_source_values = ut.get_node_distribution(self.estimator.estimators_[i], 0).reshape(-1)
    
                props_s = root_source_values
                props_s = props_s / sum(props_s)
                props_t = np.zeros(props_s.size)
                for k in range(props_s.size):
                    props_t[k] = np.sum(y_target == k) / y_target.size
                
                coeffs = np.divide(props_t, props_s)
                                
            if self.bootstrap:
                inds,oob_inds = ut._bootstrap_(y_target.size,class_wise=True,y=y_target)
            else:
                inds = np.arange(y_target.size)
    
            self.estimators_[i]._ser(X_target[inds],y_target[inds],node=0,original_ser=original_ser,
                            no_red_on_cl=no_red_on_cl,cl_no_red=cl_no_red,no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext,ext_cond=ext_cond,
                            leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,
                            Nkmin=Nkmin,max_depth=max_depth)
            
            rf_out.estimators_[i] = self.estimators_[i].estimator
            

        self.estimator = rf_out
        
        return self.estimator
    
    def _strut_rf(self,X_target,y_target,no_prune_on_cl=False,cl_no_prune=None,adapt_prop=False,
          coeffs=[1, 1],use_divergence=True,measure_default_IG=True,min_drift=None,max_drift=None,no_prune_with_translation=True,
          leaf_loss_quantify=False,leaf_loss_threshold=None,root_source_values=None,Nkmin=None):
                    
        rf_out = copy.deepcopy(self.source_model)
        
        for i in range(self.rf_size):
    
            if adapt_prop or leaf_loss_quantify:
            
                Nkmin = sum(y_target == cl_no_prune )
                root_source_values = ut.get_node_distribution(self.estimator.estimators_[i], 0).reshape(-1)
            
                props_s = root_source_values
                props_s = props_s / sum(props_s)
                props_t = np.zeros(props_s.size)
                
                for k in range(props_s.size):
                    props_t[k] = np.sum(y_target == k) / y_target.size
                    
                coeffs = np.divide(props_t, props_s)
    
                self.estimators_[i]._strut(
                      X_target,
                      y_target,
                      node=0,
                      no_prune_on_cl=no_prune_on_cl,
                      cl_no_prune=cl_no_prune,
                      adapt_prop=adapt_prop,                  
                      coeffs=coeffs,
                      use_divergence=use_divergence,
                      measure_default_IG=measure_default_IG,no_prune_with_translation=no_prune_with_translation,
                      leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold, 
                      root_source_values=root_source_values,Nkmin=Nkmin)                
                rf_out.estimators_[i] = self.estimators_[i].estimator
                
            else:
                self.estimators_[i]._strut(
                      X_target,
                      y_target,
                      node=0,
                      no_prune_on_cl=no_prune_on_cl,
                      cl_no_prune=cl_no_prune,               
                      use_divergence=use_divergence,
                      measure_default_IG=measure_default_IG,no_prune_with_translation=no_prune_with_translation,
                      root_source_values=root_source_values,Nkmin=Nkmin) 
                rf_out.estimators_[i] = self.estimators_[i].estimator
                
        self.estimator = rf_out
        
        return self.estimator

            
            


