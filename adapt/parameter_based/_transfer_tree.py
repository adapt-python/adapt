#from adapt.utils import (check_arrays,set_random_seed,check_estimator)
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score as _auc_

from adapt.base import BaseAdaptEstimator, make_insert_doc
from adapt.utils import check_arrays, set_random_seed, check_estimator, check_fitted_estimator

import adapt._tree_utils as ut


# @make_insert_doc(supervised=True)
class TransferTreeClassifier(BaseAdaptEstimator):
    """
    TransferTreeClassifier: Modify a source Decision tree on a target dataset.

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
        
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
        
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.
        
    Attributes
    ----------
    estimator_ : sklearn DecsionTreeClassifier
        Transferred decision tree classifier using target data.
        
    parents : numpy array of int.
        
    bool_parents_lr : numpy array of {-1,0,1} values.
        
    paths : numpy array of int arrays.
        
    rules : numpy array of 3-tuple arrays.
        
    depths : numpy array of int.
        
        
    Examples
    --------
    >>> from adapt.utils import make_classification_da
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from adapt.parameter_based import TransferTreeClassifier
    >>> Xs, ys, Xt, yt = make_classification_da()
    >>> src_model = DecisionTreeClassifier().fit(Xs, ys)
    >>> src_model.score(Xt, yt)
    0.62
    >>> tgt_model = TransferTreeClassifier(src_model)
    >>> tgt_model.fit(Xt[[1, -1]], yt[[1, -1]])
    >>> tgt_model.score(Xt, yt)
    0.92

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
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
               
        if not hasattr(estimator, "tree_"):
            raise ValueError("`estimator` argument has no ``tree_`` attribute, "
                                "please call `fit` on `estimator` or use "
                                "another estimator as `DecisionTreeClassifier`.")
        
        estimator = check_fitted_estimator(estimator)
        
        super().__init__(estimator=estimator,
                         Xt=Xt,
                         yt=yt,
                         copy=copy,
                         verbose=verbose,
                         random_state=random_state,
                         algo=algo,
                         **params)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
        

        
        self.parents = np.zeros(estimator.tree_.node_count,dtype=int)
        self.bool_parents_lr = np.zeros(estimator.tree_.node_count,dtype=int)
        self.rules = np.zeros(estimator.tree_.node_count,dtype=object)
        self.paths = np.zeros(estimator.tree_.node_count,dtype=object)
        self.depths = np.zeros(estimator.tree_.node_count,dtype=int)

        #Init. meta params
        self._compute_params()
        
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
            Not used. Here for sklearn compatibility.

        Returns
        -------
        self : returns an instance of self
        """
        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)

        self._modify_tree(self.estimator_, Xt, yt, **fit_params)
        
        return self

    
    def _modify_tree(self, dtree, X, y, **fit_params):
        
        # Aiguillage
        if self.algo == "src" or self.algo == "source":
            return 0
        elif self.algo == "tgt" or self.algo == "trgt" or self.algo == "target":
            return self._retrain(X,y, **fit_params)
        elif self.algo == "" or self.algo == "relab" or self.algo == "relabel":
            return self._relab(X, y, **fit_params)
        elif self.algo == "ser":
            return self._ser(X, y, **fit_params)
        
        elif self.algo == "strut":
            return self._strut(X, y, **fit_params)
        
        elif hasattr(self.algo, "__call__"):
            return self.algo(dtree, X, y, **fit_params)

    ### @@@ ###

    ###########
    
    def _compute_params(self,node=0):
        #Tree_ = self.estimator_.tree_
      
        if node == 0 :
            #default values
            self.parents[0] = -1
            self.rules[0] = (np.array([]),np.array([]),np.array([]))
            self.paths[0] = np.array([])
        else:
            parent,b = ut.find_parent(self.estimator_, node)
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
            new_f[-1] = self.estimator_.tree_.feature[parent]
            new_t[-1] = self.estimator_.tree_.threshold[parent]
            new_b[-1] = b
            self.rules[node] = (new_f,new_t,new_b)

        if self.estimator_.tree_.feature[node] != -2:
            child_l = self.estimator_.tree_.children_left[node]
            child_r = self.estimator_.tree_.children_right[node]
            dl = self._compute_params(node=child_l)
            dr = self._compute_params(node=child_r)
            return max(dl,dr)
        else:
            return self.depths[node]
            

                
    def _update_split(self,node,feature,threshold):
  
        #Tree_ = self.estimator_.tree_
        self.estimator_.tree_.feature[node] = feature
        self.estimator_.tree_.threshold[node] = threshold

        for k in ut.sub_nodes(self.estimator_.tree_, node)[1:]:
            ind_ = list(self.paths[k]).index(node)
            (p,t,b) = self.rules[k]
            (p[ind_],t[ind_]) =  (feature,threshold)
            self.rules[k] = (p,t,b)
             
        return node
    
    def _cut_leaf(self,node,leaf_value=None):

        #dTree = self.estimator_
        dic = self.estimator_.tree_.__getstate__().copy()
        dic_old = dic.copy()
        size_init = self.estimator_.tree_.node_count

        node_to_rem = ut.sub_nodes(self.estimator_.tree_, node)[1:]

        inds = list(set(np.arange(size_init)) - set(node_to_rem))
        
        dic['capacity'] = self.estimator_.tree_.capacity - len(node_to_rem)
        dic['node_count'] = self.estimator_.tree_.node_count - len(node_to_rem)
        
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

        (Tree, (n_f, n_c, n_o), b) = self.estimator_.tree_.__reduce__()
        del dic_old
        del self.estimator_.tree_

        self.estimator_.tree_ = Tree(n_f, n_c, n_o)
        self.estimator_.tree_.__setstate__(dic)
        
        self.estimator_.tree_.max_depth = max_d
        return inds.index(node)
    
    def _cut_left_right(self,node,lr):   
        
        #dTree = self.estimator_
        if lr == 1:
            cut_leaf = self._cut_leaf(self.estimator_.tree_.children_right[node])
            node = self.parents[cut_leaf]
            repl_node = self.estimator_.tree_.children_left[node]
            
        elif lr == -1:
            cut_leaf = self._cut_leaf(self.estimator_.tree_.children_left[node])
            node = self.parents[cut_leaf]
            repl_node = self.estimator_.tree_.children_right[node]
        
        dic = self.estimator_.tree_.__getstate__().copy()
        size_init = self.estimator_.tree_.node_count
        node_to_rem = [node,cut_leaf]
        inds = list(set(np.arange(size_init)) - set(node_to_rem))
        
        p, b = self.parents[node],self.bool_parents_lr[node]
  
        dic['capacity'] = self.estimator_.tree_.capacity - len(node_to_rem)
        dic['node_count'] = self.estimator_.tree_.node_count - len(node_to_rem)

        if p != -1 :
            if b == 1:
                dic['nodes']['right_child'][p] = repl_node
            elif b == -1:
                dic['nodes']['left_child'][p] = repl_node
            else:
                print('Error : need node direction with regard to its parent.')
            
        self.parents[repl_node] = p
        self.bool_parents_lr[repl_node] = b
        
        for k in ut.sub_nodes(self.estimator_.tree_, repl_node):
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

        (Tree, (n_f, n_c, n_o), b) = self.estimator_.tree_.__reduce__()
        del self.estimator_.tree_
        del dic_old
    
        self.estimator_.tree_ = Tree(n_f, n_c, n_o)
        self.estimator_.tree_.__setstate__(dic)

        self.estimator_.tree_.max_depth = max_d
        
        return inds.index(repl_node)

    def _extend(self,node,subtree):
        """adding tree tree2 to leaf f of tree tree1"""
        
        #tree1 = self.estimator_.tree_
        tree2 = subtree.tree_
        size_init = self.estimator_.tree_.node_count
        
        dic = self.estimator_.tree_.__getstate__().copy()
        dic2 = tree2.__getstate__().copy()
        size2 = tree2.node_count
        
        size_init = self.estimator_.tree_.node_count

        if self.depths[node] + dic2['max_depth'] > dic['max_depth']:
            dic['max_depth'] = self.depths[node] + tree2.max_depth
        
        dic['capacity'] = self.estimator_.tree_.capacity + tree2.capacity - 1
        dic['node_count'] = self.estimator_.tree_.node_count + tree2.node_count - 1
        
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

        (Tree, (n_f, n_c, n_o), b) = self.estimator_.tree_.__reduce__()

        self.estimator_.tree_ = Tree(n_f, n_c, n_o)
        self.estimator_.tree_.__setstate__(dic)
        del dic2
        del tree2

        try:
            self.estimator_.tree_.value[size_init:, :, subtree.classes_.astype(int)] = subtree.tree_.value[1:, :, :]
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
        self.estimator_.max_depth = self.estimator_.tree_.max_depth

        return node

    def _force_coherence(self,rule,node=0,Translate=False,indexes_nodes=list(),drifts=list(),auto_drift=True):
              
        #dtree = self.estimator_
        D_MARGIN = 1
        if Translate and not auto_drift:
            if len(indexes_nodes) != len(drifts):
                print('Error in parameter size for drifts')
                return node
            else:
                for k,n in enumerate(indexes_nodes):
                    self.updateSplit(n,self.estimator_.tree_.feature[n],self.estimator_.tree_.threshold[n]+drifts[k])
        
        phis,ths,bs = rule
        non_coherent_sense = 0
        
        phi,th = self.estimator_.tree_.feature[node],self.estimator_.tree_.threshold[node]
        
        if phi != -2:
        #if it is not a leaf
            coh,non_coherent_sense = ut.coherent_new_split(phi,th,rule)
                        
            if not coh:
                if Translate :
                    if auto_drift:
                        try:
                            n_feat = self.estimator_.n_features_
                        except:
                            n_feat = self.estimator_.n_features_in_
                        b_infs,b_sups = ut.bounds_rule(rule, n_feat)

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
                        phi,th = self.estimator_.tree_.feature[node],self.estimator_.tree_.threshold[node]
                        coh,non_coherent_sense = ut.coherent_new_split(phi,th,rule)
    
            node_l = self.estimator_.tree_.children_left[node]   
            rule_l = self.rules[node_l]
            if self.estimator_.tree_.feature[node_l] != -2 :
                node_l = self._force_coherence(rule_l,node=node_l,Translate=Translate,
                                             indexes_nodes=indexes_nodes,drifts=drifts,auto_drift=auto_drift)

            node = self.parents[node_l]

            node_r = self.estimator_.tree_.children_right[node]  
            rule_r = self.rules[node_r]
            if self.estimator_.tree_.feature[node_r] != -2 :
                node_r = self._force_coherence(rule_r,node=node_r,Translate=Translate,
                                             indexes_nodes=indexes_nodes,drifts=drifts,auto_drift=auto_drift)
            node = self.parents[node_r]  
        
            return node
            
    ### @@@ ###
    
    ###########

    def updateSplit(self,node,feature,threshold):
        """
        Update the (feature,threshold) split for a given node.
        
        Parameters
        ----------
        node : int
            Node to update.
            
        feature : int
            New node feature.
            
        threshold : float
            New node threshold.
        
        """
        return self._update_split(node,feature,threshold)
        
    def updateValue(self,node,values):
        """
        Update class values for a given node.
        
        Parameters
        ----------
        node : int
            Node to update.
            
        values : numpy array of float
            Class values to affect at node.
        
        """
        #Tree_ = self.estimator_.tree_
        self.estimator_.tree_.value[node] = values
        self.estimator_.tree_.impurity[node] = ut.GINI(values)
        self.estimator_.tree_.n_node_samples[node] = np.sum(values)
        self.estimator_.tree_.weighted_n_node_samples[node] = np.sum(values)
        return node
    
    def swap_subtrees(self,node1,node2):
        """
        Swap respective sub-trees between two given nodes.
        
        Each node must not be a sub-node of the other.
        Update the (feature,threshold) split for a given node.
        
        Parameters
        ----------
        node1 : int
            Node to swap.
            
        node2 : int
            Node to swap.            
        """
        #Check sub-nodes :
        if node1 == node2:
            print('Warning : same node given twice.')
            return 0
        
        if node2 in ut.sub_nodes(self.estimator_.tree_, node1)[1:]:
            print('Error : node2 is a sub-node of node1.')
            return 0

        if node1 in ut.sub_nodes(self.estimator_.tree_, node2)[1:]:
            print('Error : node1 is a sub-node of node2.')
            return 0

        p1,b1 = self.parents[node1], self.bool_parents_lr[node1]
        p2,b2 = self.parents[node2], self.bool_parents_lr[node2]
        
        if b1 == -1:
            self.estimator_.tree_.children_left[p1] = node2
        elif b1 == 1:
            self.estimator_.tree_.children_right[p1] = node2

        if b2 == -1:
            self.estimator_.tree_.children_left[p2] = node1
        elif b2 == 1:
            self.estimator_.tree_.children_right[p2] = node1
            
        self.parents[node2] = p1
        self.bool_parents_lr[node2] = b1            
        self.parents[node1] = p2
        self.bool_parents_lr[node1] = b2

        d1 = self._compute_params(node=node1) 
        d2 = self._compute_params(node=node2) 
        
        self.estimator_.tree_.max_depth = max(d1,d2)
        self.estimator_.max_depth = self.estimator_.tree_.max_depth
        
        return 1
        
    def prune(self,node,include_node=False,lr=0,leaf_value=None):
        """
        Pruning the corresponding sub-tree at a given node.
        
        If `include_node` is `False`, replaces the node by a leaf (with values `leaf_values` if provided).
        If `include_node` is `True`, prunes the left (`lr=-1`) or (`lr=1`) child sub-tree and
        replaces the given node by the other sub-tree.
        
        Parameters
        ----------
        node : int
            Node to prune.
            
        include_node : boolean (default=False)
            Type of pruning to apply.
        
        lr : float
            Direction of pruning if `include_node` is `True`.
            Must be either -1 (left) or 1 (right) in this case.
            
        leaf_value : numpy array of float (default=None)
            If `include_node` is `False`, affects these values to the created leaf.          
        """
        if include_node:
            n = self._cut_left_right(node,lr)
        else:
            n = self._cut_leaf(node,leaf_value=leaf_value)
        return n

    def extend(self,node,subtree):
        """
        Extend the underlying decision tree estimator by a sub-tree at a given node.
        
        Parameters
        ----------
        node : int
            Node to update.
            
        subtree : DecisionTreeClassifier.        
        """
        n = self._extend(node,subtree)
        return n

    ### @@@ ###

    ###########
    
    def _retrain(self,X_target, Y_target):
        self.estimator_.fit(X_target, Y_target)


    def _relab(self, X_target_node, Y_target_node, node=0):
        
        #Tree_ = self.estimator_.tree_
        classes_ = self.estimator_.classes_
        
        current_class_distribution = ut.compute_class_distribution(classes_, Y_target_node)
        self.updateValue(node,current_class_distribution)
        
        bool_test = X_target_node[:, self.estimator_.tree_.feature[node]] <= self.estimator_.tree_.threshold[node]
        not_bool_test = X_target_node[:, self.estimator_.tree_.feature[node]] > self.estimator_.tree_.threshold[node]

        ind_left = np.where(bool_test)[0]
        ind_right = np.where(not_bool_test)[0]

        X_target_node_left = X_target_node[ind_left]
        Y_target_node_left = Y_target_node[ind_left]

        X_target_node_right = X_target_node[ind_right]
        Y_target_node_right = Y_target_node[ind_right]
        
        if self.estimator_.tree_.feature[node] != -2:
            self._relab(X_target_node_left,Y_target_node_left,node=self.estimator_.tree_.children_left[node])
            self._relab(X_target_node_right,Y_target_node_right,node=self.estimator_.tree_.children_right[node])

        return node


    def _ser(self,X_target_node,y_target_node,node=0,original_ser=True,
             no_red_on_cl=False,cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,ext_cond=None,
             leaf_loss_quantify=False,leaf_loss_threshold=None,coeffs=[1,1],root_source_values=None,Nkmin=None,max_depth=None):
        
        #Tree_ = self.estimator_.tree_
        if (no_red_on_cl or no_ext_on_cl ) and node == 0:
            if Nkmin is None:
                if no_ext_on_cl:
                    Nkmin = sum(y_target_node == cl_no_ext )
                if no_red_on_cl:
                    Nkmin = sum(y_target_node == cl_no_red )
            if root_source_values is None:
                root_source_values = ut.get_node_distribution(self.estimator_, 0).reshape(-1)


            if coeffs is None or list(coeffs) == [1,1]:        
                props_s = root_source_values
                props_s = props_s / sum(props_s)
                props_t = np.zeros(props_s.size)
                
                for k in range(props_s.size):
                    props_t[k] = np.sum(y_target_node == k) / y_target_node.size
                    
                coeffs = np.divide(props_t, props_s)
                
        source_values = self.estimator_.tree_.value[node].copy()
        #node_source_label = np.argmax(source_values)
        maj_class = np.argmax(self.estimator_.tree_.value[node, :].copy())

        if cl_no_red is None:
            old_size_cl_no_red = 0
        else:
            old_size_cl_no_red = np.sum(self.estimator_.tree_.value[node][:, cl_no_red])
            
        # Situation où il y a des restrictions sur plusieurs classes ?
        if no_red_on_cl is not None or no_ext_on_cl is not None :
            if no_ext_on_cl:
                cl = cl_no_ext[0]
            if no_red_on_cl:
                cl = cl_no_red[0]

        if (leaf_loss_quantify is True ) and ((no_red_on_cl  or  no_ext_on_cl) and maj_class == cl) and  self.estimator_.tree_.feature[node] == -2 :
            if Nkmin is None:
                Nkmin = sum(y_target_node == cl_no_red )
            ps_rf = self.estimator_.tree_.value[node,0,:]/sum(self.estimator_.tree_.value[node,0,:])
            p1_in_l = self.estimator_.tree_.value[node,0,cl]/root_source_values[cl]
            
            cond_homog_unreached = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
            cond_homog_min_label = np.argmax(np.multiply(coeffs,ps_rf)) == cl
            
        val = np.zeros((self.estimator_.n_outputs_, self.estimator_.n_classes_))

        for i in range(self.estimator_.n_classes_):
            val[:, i] = list(y_target_node).count(i)
        
        self.updateValue(node,val)
        
        if self.estimator_.tree_.feature[node]== -2:
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
                            
                            ut.add_to_parents(self.estimator_, node, source_values) 
                            if no_red_on_cl:
                                bool_no_red = True
                                        
    
                # No red protection with values / used to flag tree parts concerned by pruning restrictions
                if no_red_on_cl and y_target_node.size == 0 and old_size_cl_no_red > 0 and maj_class in cl_no_red:
                    
                    if leaf_loss_quantify :
                        if cond_homog_unreached and cond_homog_min_label :
                            self.updateValue(node,source_values)
                            
                            ut.add_to_parents(self.estimator_, node, source_values) 
                            bool_no_red = True
                    else:
                        self.updateValue(node,source_values)

                        ut.add_to_parents(self.estimator_, node, source_values) 
                        bool_no_red = True

                return node,bool_no_red
        
        """ From here it cannot be a leaf """
        ### Left / right target computation ###
        bool_test = X_target_node[:, self.estimator_.tree_.feature[node]] <= self.estimator_.tree_.threshold[node]
        not_bool_test = X_target_node[:, self.estimator_.tree_.feature[node]] > self.estimator_.tree_.threshold[node]

        ind_left = np.where(bool_test)[0]
        ind_right = np.where(not_bool_test)[0]

        X_target_node_left = X_target_node[ind_left]
        y_target_node_left = y_target_node[ind_left]

        X_target_node_right = X_target_node[ind_right]
        y_target_node_right = y_target_node[ind_right]

        if original_ser:

            new_node_left,bool_no_red_l = self._ser(X_target_node_left,y_target_node_left,node=self.estimator_.tree_.children_left[node],original_ser=True,max_depth=max_depth)
            node = self.parents[new_node_left]

            new_node_right,bool_no_red_r = self._ser(X_target_node_right,y_target_node_right,node=self.estimator_.tree_.children_right[node],original_ser=True,max_depth=max_depth)
            node = self.parents[new_node_right]
                                  
        else:
            new_node_left,bool_no_red_l = self._ser(X_target_node_left,y_target_node_left,node=self.estimator_.tree_.children_left[node],original_ser=False,
                                               no_red_on_cl=no_red_on_cl,cl_no_red=cl_no_red,no_ext_on_cl=no_ext_on_cl,cl_no_ext=cl_no_ext,ext_cond=ext_cond,
                                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,
                                               Nkmin=Nkmin,max_depth=max_depth)


            node = self.parents[new_node_left]

            new_node_right,bool_no_red_r = self._ser(X_target_node_right,y_target_node_right,node=self.estimator_.tree_.children_right[node],original_ser=False,
                                               no_red_on_cl=no_red_on_cl,cl_no_red=cl_no_red,no_ext_on_cl=no_ext_on_cl,cl_no_ext=cl_no_ext,ext_cond=ext_cond,
                                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,coeffs=coeffs,root_source_values=root_source_values,
                                               Nkmin=Nkmin,max_depth=max_depth)

            node = self.parents[new_node_right]

        if original_ser:
            bool_no_red = False
        else:
            bool_no_red = bool_no_red_l or bool_no_red_r

        le = ut.leaf_error(self.estimator_.tree_, node)
        e = ut.error(self.estimator_.tree_, node)

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

        if self.estimator_.tree_.feature[node] != -2:

            if original_ser:
                if ind_left.size == 0:
                    node = self.prune(node,include_node=True,lr=-1) 
                    
                if ind_right.size == 0:
                    node = self.prune(node,include_node=True,lr=1)
            else:
                if no_red_on_cl:
                    if ind_left.size == 0 and np.sum(self.estimator_.tree_.value[self.estimator_.tree_.children_left[node]]) == 0:
                        node = self.prune(node,include_node=True,lr=-1) 
                        
                    if ind_right.size == 0 and np.sum(self.estimator_.tree_.value[self.estimator_.tree_.children_right[node]]) == 0:
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
                
#        Tree_ = self.estimator_.tree_

        if (no_prune_on_cl or leaf_loss_quantify or adapt_prop) and node == 0:

            if Nkmin is None:
                Nkmin = sum(Y_target_node == cl_no_prune )
            if root_source_values is None:
                root_source_values = ut.get_node_distribution(self.estimator_, 0).reshape(-1)

            if coeffs is None or list(coeffs) == [1,1]:        
                props_s = root_source_values
                props_s = props_s / sum(props_s)
                props_t = np.zeros(props_s.size)
                
                for k in range(props_s.size):
                    props_t[k] = np.sum(Y_target_node == k) / Y_target_node.size
                    
                coeffs = np.divide(props_t, props_s)
            
        feature_ = self.estimator_.tree_.feature[node]
        classes_ = self.estimator_.classes_
        threshold_ = self.estimator_.tree_.threshold[node]
            
        old_threshold = threshold_.copy()
        maj_class = np.argmax(self.estimator_.tree_.value[node, :].copy())
        
        if min_drift is None or max_drift is None:
            try:
                n_feat = self.estimator_.n_features_
            except:
                n_feat = self.estimator_.n_features_in_
            min_drift = np.zeros(n_feat)
            max_drift = np.zeros(n_feat)

        current_class_distribution = ut.compute_class_distribution(classes_, Y_target_node)
        is_reached = (Y_target_node.size > 0)
        no_min_instance_targ = False
        
        if no_prune_on_cl:
            no_min_instance_targ = (sum(current_class_distribution[cl_no_prune]) == 0 )
            is_instance_cl_no_prune = np.sum(self.estimator_.tree_.value[node, :,cl_no_prune].astype(int))

        # If it is a leaf :
        if self.estimator_.tree_.feature[node] == -2:
            """ When to apply UpdateValue """
            if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) :

                ps_rf = self.estimator_.tree_.value[node,0,:]/sum(self.estimator_.tree_.value[node,0,:])
                p1_in_l = self.estimator_.tree_.value[node,0,cl_no_prune]/root_source_values[cl_no_prune]
                cond1 = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
                cond2 = np.argmax(np.multiply(coeffs,ps_rf)) == cl_no_prune
                
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune:
                if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) and not(cond1 and cond2):
                    self.updateValue(node,current_class_distribution)
                    return node
                else:
                    return node
            else:
                #self.estimator_.tree_.value[node] = current_class_distribution
                """ UpdateValue """
                self.updateValue(node,current_class_distribution)
                return node

        # Only one class remaining in target :
        if (current_class_distribution > 0).sum() == 1:
            """ When to apply Pruning and how if not """
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune :
                bool_subleaf_noprune = True
                if leaf_loss_quantify:
                    bool_subleaf_noprune = ut.contain_leaf_to_not_prune(self.estimator_,cl=cl_no_prune,node=node,Nkmin=Nkmin,
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
                    bool_subleaf_noprune = ut.contain_leaf_to_not_prune(self.estimator_,cl=cl_no_prune,node=node,
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
            Q_source_l, Q_source_r = ut.get_children_distributions(self.estimator_,node)

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
    
    
        Q_source_parent = ut.get_node_distribution(self.estimator_,node)
            
                                                
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
            child_l = self.estimator_.tree_.children_left[node]
            child_r = self.estimator_.tree_.children_right[node]
            self.swap_subtrees(child_l,child_r)

        # For No Prune coherence
        ecart = self.estimator_.tree_.threshold[node] - old_threshold
        
        if self.estimator_.tree_.threshold[node] > old_threshold:
            if ecart > max_drift[self.estimator_.tree_.feature[node]] :
                max_drift[self.estimator_.tree_.feature[node]] = ecart
        if self.estimator_.tree_.threshold[node] < old_threshold:
            if ecart < min_drift[self.estimator_.tree_.feature[node]] :
                min_drift[self.estimator_.tree_.feature[node]] = ecart

        if self.estimator_.tree_.children_left[node] != -1:

            threshold = self.estimator_.tree_.threshold[node]
            index_X_child_l = X_target_node[:, feature_] <= threshold
            X_target_child_l = X_target_node[index_X_child_l, :]
            Y_target_child_l = Y_target_node[index_X_child_l]

            node_l = self._strut(X_target_child_l,Y_target_child_l,
                          node=self.estimator_.tree_.children_left[node],no_prune_on_cl=no_prune_on_cl,cl_no_prune=cl_no_prune,
                          adapt_prop=adapt_prop,coeffs=coeffs,use_divergence=use_divergence,measure_default_IG=measure_default_IG,
                          min_drift=min_drift.copy(),max_drift=max_drift.copy(),no_prune_with_translation=no_prune_with_translation,
                          leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,root_source_values=root_source_values,Nkmin=Nkmin)
        
            node = self.parents[node_l]

        if self.estimator_.tree_.children_right[node] != -1:

            threshold = self.estimator_.tree_.threshold[node]
            index_X_child_r = X_target_node[:, feature_] > threshold
            X_target_child_r = X_target_node[index_X_child_r, :]
            Y_target_child_r = Y_target_node[index_X_child_r]
            
            node_r = self._strut(X_target_child_r,Y_target_child_r,
                          node=self.estimator_.tree_.children_right[node],no_prune_on_cl=no_prune_on_cl,cl_no_prune=cl_no_prune,
                          adapt_prop=adapt_prop,coeffs=coeffs,use_divergence=use_divergence,measure_default_IG=measure_default_IG,
                          min_drift=min_drift.copy(),max_drift=max_drift.copy(),no_prune_with_translation=no_prune_with_translation,
                          leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,root_source_values=root_source_values,Nkmin=Nkmin)

            node = self.parents[node_r]
              
        return node

        
class TransferTreeSelector(BaseAdaptEstimator):
    """
    TransferTreeSelector : Run several decision tree transfer algorithms on a target dataset and select the best one.
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 algorithms=list(),
                 list_alg_args=list(),
                 data_size_per_class=None,
                 root_source_values=None,
                 coeffs=[1,1],
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
               
        if not hasattr(estimator, "tree_"):
            raise ValueError("`estimator` argument has no ``tree_`` attribute, "
                                "please call `fit` on `estimator` or use "
                                "another estimator as `DecisionTreeClassifier`.")
        
        estimator = check_fitted_estimator(estimator)
        
        super().__init__(estimator=estimator,
                         Xt=Xt,
                         yt=yt,
                         copy=copy,
                         verbose=verbose,                       
                         **params)

        
        if len(algorithms) == 0:
            print('Warning : empty list of methods. Default are Source and Target models.')
            self.algorithms = ['src','trgt']
            self.list_alg_args = [{},{}]
        else:
            self.algorithms = algorithms
            self.list_alg_args = list_alg_args
            
        if len(self.list_alg_args) == 0 and len(self.algorithms) != 0 :
            self.list_alg_args = list(np.repeat({},len(self.algorithms)))
        
        self.n_methods = len(self.algorithms)
        self.scores = np.zeros(self.n_methods)
        
        self.best_score = 0
        self.best_index = -1
        
        self.data_size_per_class = data_size_per_class
        self.root_source_values = root_source_values
        self.coeffs = coeffs
        
        self.transferred_models = list()
        
        for algo in self.algorithms:

            self.transferred_models.append(TransferTreeClassifier(estimator=self.estimator,Xt=self.Xt,yt=self.yt,algo=algo,copy=self.copy))
            
      
    def fit(self, Xt=None, yt=None, **fit_params):

        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
        
        for k,algo in enumerate(self.algorithms):
            kwargs = self.list_alg_args[k]
            
            if 'Nkmin' not in kwargs.keys():
                if 'cl_no_red' in kwargs.keys():
                    cl = kwargs['cl_no_red']
                    self.Nkmin = self.data_size_per_class[cl]
                    kwargs['Nkmin'] = self.Nkmin
                elif 'cl_no_prune' in kwargs.keys():
                    cl = kwargs['cl_no_prune']
                    self.Nkmin = self.data_size_per_class[cl]
                    kwargs['Nkmin'] = self.Nkmin
                elif 'cl_no_ext' in kwargs.keys():
                    cl = kwargs['cl_no_ext']
                    self.Nkmin = self.data_size_per_class[cl]
                    kwargs['Nkmin'] = self.Nkmin
            if ('leaf_loss_quantify' in kwargs.keys() and kwargs['leaf_loss_quantify']) or ('adapt_prop' in kwargs.keys() and kwargs['adapt_prop']):
                kwargs['root_source_values'] = self.root_source_values
                kwargs['coeffs'] = self.coeffs

            
            #self.transferred_models[k].fit(Xt=Xt, yt=yt,Nkmin=self.Nkmin,root_source_values=self.root_source_values,coeffs=self.coeffs,**kwargs,**fit_params)
            self.transferred_models[k].fit(Xt=Xt, yt=yt,**kwargs,**fit_params)
            
    def select(self,Xtest=None,ytest=None,score_type="auc"):

        Xtest, ytest = self._get_target_data(Xtest, ytest)
        Xtest, ytest = check_arrays(Xtest, ytest)
        set_random_seed(self.random_state)
        
        for k in range(len(self.algorithms)):
            if score_type == "auc":
                self.scores[k] = _auc_(ytest,self.transferred_models[k].estimator_.predict_proba(Xtest)[:,1]) 
            else:
                self.scores[k] = self.transferred_models[k].score(Xtest,ytest) 
        
        self.best_score = np.amax(self.scores)
        self.best_index = np.argmax(self.scores)
        self.best_method = self.algorithms[self.best_index]
        
        return self.best_score, self.best_index
    
class TransferForestClassifier(BaseAdaptEstimator):
    """
    TransferForestClassifier: Modify a source Random Forest on a target dataset.
    
    Random forest classifier structure for model-based transfer algorithms.
    
    This includes several algorithms : leaves relabeling according to target data, SER and STRUT algorithms
    and various variants for target class imbalance situations.
    
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
        
    copy : boolean (default=True)
        Whether to make a copy of ``estimator`` or not.
        
    verbose : int (default=1)
        Verbosity level.
        
    random_state : int (default=None)
        Seed of random generator.
        
    Attributes
    ----------
    estimator_ : sklearn RandomForestClassifier
        Transferred random forest classifier using target data.

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
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        if not isinstance(estimator, RandomForestClassifier):
            raise ValueError("`estimator` argument must be a ``RandomForestClassifier`` instance, got %s."%str(type(estimator)))

        if not hasattr(estimator, "estimators_"):
            raise ValueError("`estimator` argument has no ``estimators_`` attribute, "
                                "please call `fit` on `estimator`.")
        
        estimator = check_fitted_estimator(estimator)
        
        super().__init__(estimator=estimator,
                         Xt=Xt,
                         yt=yt,
                         copy=copy,
                         verbose=verbose,
                         random_state=random_state,
                         algo=algo,                         
                         bootstrap=bootstrap,
                         **params)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
                
        
        self.rf_size = self.estimator_.n_estimators
        self.estimators_ = np.zeros(self.rf_size,dtype=object)

        for i in range(self.rf_size):
            self.estimators_[i] = TransferTreeClassifier(estimator = self.estimator_.estimators_[i], algo = self.algo)


    ### @@@ ###

    ###########
    
    def _modify_rf(self, rf, X, y, **fit_params):
        # Aiguillage
        if self.algo == "src" or self.algo == "source":
            return 0
        elif self.algo == "tgt" or self.algo == "trgt" or self.algo == "target":
            return self._retrain(X,y, **fit_params)
        if self.algo == "" or self.algo == "relabel":
            return self._relab_rf(X, y, **fit_params)
        elif self.algo == "ser":
            return self._ser_rf(X, y, **fit_params)        
        elif self.algo == "strut":
            return self._strut_rf(X, y, **fit_params)
        
        elif hasattr(self.algo, "__call__"):
            return self.algo(rf, X, y, **fit_params)

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

        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
            
        self._modify_rf(self.estimator_, Xt, yt, **fit_params)
        
        return self
    
    def _copy_rf(self):
        rf_out = copy.deepcopy(self.estimator)
        for k in range(self.rf_size):
            rf_out.estimators_[k] = self.estimators_[k].estimator_
        self.estimator_ = rf_out

    def _copy_dt(self):
        for k in range(self.rf_size):
            self.estimators_[k].estimator_ = self.estimator_.estimators_[k]
        
    def _retrain(self,X_target, Y_target):
        self.estimator_.fit(X_target, Y_target)
        self._copy_dt()
        
    def _relab_rf(self, X_target_node, Y_target_node):
        
        rf_out = copy.deepcopy(self.estimator)
        if self.bootstrap :             
            inds,oob_inds = ut._bootstrap_(Y_target_node.size,class_wise=True,y=Y_target_node)
            for k in range(self.rf_size):
                X_target_node_bootstrap = X_target_node[inds]
                Y_target_node_bootstrap = Y_target_node[inds]
                self.estimators_[k]._relab(X_target_node_bootstrap, Y_target_node_bootstrap, node=0)
                rf_out.estimators_[k] = self.estimators_[k].estimator_
        else:            
            for k in range(self.rf_size):
                self.estimators_[k]._relab(X_target_node, Y_target_node, node=0)
                rf_out.estimators_[k] = self.estimators_[k].estimator_
                
        self.estimator_ = rf_out
        self._copy_dt()
        self._copy_rf()
        return self.estimator_
    
    def _ser_rf(self,X_target,y_target,original_ser=True,
             no_red_on_cl=False,cl_no_red=None, no_ext_on_cl=False, cl_no_ext=None,ext_cond=None,
             leaf_loss_quantify=False,leaf_loss_threshold=None,coeffs=[1,1],root_source_values=None,Nkmin=None,max_depth=None):
        
        rf_out = copy.deepcopy(self.estimator)
        for i in range(self.rf_size):
            root_source_values = None
            coeffs = None
            Nkmin = None
            if  leaf_loss_quantify :    
                Nkmin = sum(y_target == cl_no_red )
                root_source_values = ut.get_node_distribution(self.estimator_.estimators_[i], 0).reshape(-1)
    
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
            
            rf_out.estimators_[i] = self.estimators_[i].estimator_
            
        self.estimator_ = rf_out
        self._copy_dt()
        self._copy_rf()
        
        return self.estimator_
    
    def _strut_rf(self,X_target,y_target,no_prune_on_cl=False,cl_no_prune=None,adapt_prop=False,
          coeffs=[1, 1],use_divergence=True,measure_default_IG=True,min_drift=None,max_drift=None,no_prune_with_translation=True,
          leaf_loss_quantify=False,leaf_loss_threshold=None,root_source_values=None,Nkmin=None):
                    
        rf_out = copy.deepcopy(self.estimator)
        for i in range(self.rf_size):
    
            if adapt_prop or leaf_loss_quantify:
            
                Nkmin = sum(y_target == cl_no_prune )
                root_source_values = ut.get_node_distribution(self.estimator_.estimators_[i], 0).reshape(-1)
            
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
                
            rf_out.estimators_[i] = self.estimators_[i].estimator_
                
        self.estimator_ = rf_out
        self._copy_dt()
        self._copy_rf()
        
        return self.estimator_      
       


        
class TransferForestSelector(BaseAdaptEstimator):
    """
    TransferForestSelector : Run several decision tree transfer algorithms on a target dataset and select the best one for each tree of the random forest.
    
    """
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 algorithms=list(),
                 list_alg_args=list(),
                 bootstrap=True,
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        if not isinstance(estimator, RandomForestClassifier):
            raise ValueError("`estimator` argument must be a ``RandomForestClassifier`` instance, got %s."%str(type(estimator)))

        if not hasattr(estimator, "estimators_"):
            raise ValueError("`estimator` argument has no ``estimators_`` attribute, "
                                "please call `fit` on `estimator`.")
        
        estimator = check_fitted_estimator(estimator)
        
        super().__init__(estimator=estimator,
                         Xt=Xt,
                         yt=yt,
                         copy=copy,
                         verbose=verbose,
                         random_state=random_state,                       
                         bootstrap=bootstrap,
                         **params)
        

        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
                
        
        self.rf_size = self.estimator_.n_estimators

        
        if len(algorithms) == 0:
            print('Warning : empty list of methods. Default are Source and Target models.')
            self.algorithms = ['src','trgt']
            self.list_alg_args = [{},{}]
        else:
            self.algorithms = algorithms
            self.list_alg_args = list_alg_args
            
        if len(self.list_alg_args) == 0 and len(self.algorithms) != 0 :
            self.list_alg_args = list(np.repeat({},len(self.algorithms)))
            

        self.n_methods = len(self.algorithms)
        self.scores = np.zeros(self.n_methods)
        
        self.best_score = 0
        self.best_index = -1
        
        self.transferred_models = list()
        
        for algo in self.algorithms:
            self.transferred_models.append(TransferForestClassifier(estimator=self.estimator,Xt=self.Xt,yt=self.yt,algo=algo,bootstrap=self.bootstrap,copy=self.copy))
            
        self.STRF_model = TransferForestClassifier(estimator=self.estimator,Xt=self.Xt,yt=self.yt,algo=algo,bootstrap=self.bootstrap,copy=self.copy)
        self.STRF_indexes = np.zeros(self.rf_size)

                
    def model_selection(self, Xt=None, yt=None, score_type = "auc", oob_ = False, **fit_params):

        Xt, yt = self._get_target_data(Xt, yt)
        Xt, yt = check_arrays(Xt, yt)
        set_random_seed(self.random_state)
        
        rf_out = copy.deepcopy(self.estimator)


        for k in range(self.rf_size):

            # For imbalance adaptation:
            data_size_per_class = np.zeros(rf_out.n_classes_)
            props_t = np.zeros(rf_out.n_classes_)
            
            for cl in range(rf_out.n_classes_):
                data_size_per_class[cl] = sum(yt == cl )
            props_t = data_size_per_class / yt.size
            
            root_source_values = ut.get_node_distribution(rf_out.estimators_[k], 0).reshape(-1)

            props_s = root_source_values
            props_s = props_s / sum(props_s)     
            coeffs = np.divide(props_t, props_s)

            TTS = TransferTreeSelector(estimator=self.estimator_.estimators_[k],algorithms=self.algorithms,
                                       list_alg_args=self.list_alg_args,
                                       data_size_per_class=data_size_per_class,
                                       root_source_values=root_source_values,coeffs=coeffs)
            
            if self.bootstrap:
                inds, oob_inds = ut._bootstrap_(yt.size,class_wise=True,y=yt)
                TTS.fit(Xt[inds],yt[inds],**fit_params)         
                
                if len(set(yt[oob_inds])) == 1:
                    print('Warning: Only one class in OOB samples.')
                    oob_ = False
                if oob_:
                    score, index = TTS.select(Xtest=Xt[oob_inds],ytest=yt[oob_inds],score_type=score_type)
                else:
                    score, index = TTS.select(Xtest=Xt[inds],ytest=yt[inds],score_type=score_type)
            else:
                TTS.fit(Xt,yt,**fit_params)                
                score, index = TTS.select(Xtest=Xt,ytest=yt,score_type=score_type)                 

            self.STRF_indexes[k] = index
            
            self.STRF_model.estimators_[k] = TTS.transferred_models[index]
            
            for j,m in enumerate(self.transferred_models):
                #rf_out_alg = copy.deepcopy(rf_out)
                m.estimators_[k] = TTS.transferred_models[j]
                m.estimator_.estimators_[k] = TTS.transferred_models[j].estimator_
                #m.estimator_ = rf_out_alg
                
            rf_out.estimators_[k] = TTS.transferred_models[index].estimator_
        
        self.STRF_model.estimator_ = rf_out
        self.estimator_ = rf_out
        
        return self.STRF_indexes


