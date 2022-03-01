"""
    SER :
    fusionDT,add_to_parents,find_parent,error,leaf_error
    cut_from_left_right,cut_into_leaf
    
    STRUT:
    extract_rule,Force_Coherence,contain_leaf_not_prune
    compute_class_distr,get_children_distr, get_node_distr
    Gini, DG, threshold_selection, compute_Q_children_target
    cut_from_left_right,cut_into_leaf
    
    Fichier tree_utils : error, leaf_error, DG, etc...
"""

from adapt.utils import (check_arrays,
                         set_random_seed,
                         check_estimator)

import tree_utils as ut

class TransferTreeClassifier:
    """
    TransferTreeClassifier
    
    TTC do that and that ...
    
    Parameters
    ----------    
    estimator : sklearn DecsionTreeClassifier (default=None)
        ...
        
    algo : str or callable (default="ser")
        ...
        
    (pas la peine de commenter Xt, yt, copy, verbose et random_state)
        
    Attributes
    ----------
    estimator_ : Same class as estimator
        Fitted Estimator.
        
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
    .. [1] `[1] < le lien vers l'article >`_ le nom \
des auteurs. " Le titre de l'article ". In conference, année.
    """
    
    
    def __init__(self,
                 estimator=None,
                 Xt=None,
                 yt=None,
                 algo="ser",
                 copy=True,
                 verbose=1,
                 random_state=None,
                 **params):
        
        if not hasattr(estimator, "tree_"):
            raise NotFittedError("`estimator` argument has no ``tree_`` attribute, "
                                 "please call `fit` on `estimator` or use "
                                 "another estimator.")
        
        self.parents = np.zeros(estimator.tree_.node_count,dtype=int)
        self.bool_parents_lr = np.zeros(estimator.tree_.node_count,dtype=int)
        self.rules = np.zeros(estimator.tree_.node_count,dtype=object)
        self.depths = np.zeros(estimator.tree_.node_count,dtype=int)
        
        self.estimator = estimator
        self.Xt = Xt
        self.yt = yt
        self.algo = algo
        self.copy = copy
        self.verbose = verbose
        self.random_state = random_state
        self.params = params

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
            Arguments for the estimator.

        Returns
        -------
        self : returns an instance of self
        """
        
        if self.estimator is None:
        #Pas d'arbre source
        
        if self.estimator.node_count == 0:
        #Arbre vide
        
        set_random_seed(self.random_state)
        Xt, yt = check_arrays(Xt, yt)
        
        self.estimator_ = check_estimator(self.estimator,
                                          copy=self.copy,
                                          force_copy=True)
        
        Tree_ = self.estimator_.tree_
        
        Tree_ = self._modify_tree(Tree_, Xt, yt)
        
        return self
        
    """
        def _build_tree(self, X, y):
        # Construit un arbre et le fit sur X, y
        pass
        
        
        def _prune(self, tree):
        # Prune tree à sa racine
        pass
    """

    def _check_coherence_values(self):
    
    def _get_(self):
        #    extract_rule,find_parent,contain_leaf_not_prune,compute_class_distr,
        #    get_children_distr, get_node_distr
        
    def _modify_tree(self, tree, X, y):
        
        # Aiguillage
        if self.algo == "" or "relabel":
            return self._ser(X, y)
        elif self.algo == "ser":
            return self._ser(X, y)
        
        elif self.algo == "strut":
            return self._strut(X, y)
        
        elif hasattr(self.algo, "__call__"):
            return self.algo(tree, X, y)

    ### @@@ ###

    ###########
    def _compute_params(self,node=0):
        Tree_ = self.estimator.tree_
        
        if node == 0 :
            #default values
            self.parents[0] = -1
            self.rules[0] = (np.array([]),np.array([]),np.array([]))
        else:
            parent,b = ut.find_parent(self.estimator, node)
            self.parents[node] = parent
            self.bool_parents_lr[node] = parent
            self.depths[node] = self.depths[parent]+1
            
            (features,thresholds,bs) = self.rules[parent]
            new_f=np.zeros(features.size+1)
            new_t=np.zeros(thresholds.size+1)
            new_b=np.zeros(bs.size+1)
            new_f[:-1] = features
            new_t[:-1] = thresholds
            new_b[:-1] = bs
            new_f[-1] = Tree_.feature[parent]
            new_t[-1] = Tree_.threshold[parent]
            new_b[-1] = b
            self.rules[node] = (new_f,new_t,new_b)

            if Tree_.feature[node] != -2:
                child_l = Tree_.children_left[node]
                child_r = Tree_.children_right[node]
                self._compute_params(child_l)
                self._compute_params(child_r)
                
    def _update_split(self,node,feature,threshold):
        # Juste changer la liste des rules concernées
        Tree_ = self.estimator.tree_
        Tree_.feature[node] = feature
        Tree_.threshold[node] = threshold
    
        (p,t,b) = self.rules[node][self.depths[node]]
        self.rules[node][self.depths[node]] = (feature,threshold,b)
        return node
    
    def _cut_leaf(self,node,leaf_value=None):
        # Changer array parents + rules + depths
        dic = dTree.tree_.__getstate__().copy()
        size_init = dTree.tree_.node_count

        #node_to_rem = list()
        #node_to_rem = node_to_rem + sub_nodes(dTree.tree_, node)[1:]
        #node_to_rem = list(set(node_to_rem))
        node_to_rem = sub_nodes(dTree.tree_, node)[1:]
        
        #inds = list(set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
        inds = list(set(np.arange(size_init)) - set(node_to_rem))
        #depths = depth_array(dTree, inds)
        #dic['max_depth'] = np.max(depths)
        
        dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
        dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)
        
        dic['nodes']['feature'][node] = -2
        dic['nodes']['left_child'][node] = -1
        dic['nodes']['right_child'][node] = -1
        
        dic_old = dic.copy()
        left_old = dic_old['nodes']['left_child']
        right_old = dic_old['nodes']['right_child']
        dic['nodes'] = dic['nodes'][inds]
        dic['values'] = dic['values'][inds]
        
        self.parents = self.parents[inds]
        self.bool_parents_lr = self.bool_parents_lr[inds]
        self.rules = self.rules[inds]
        self.depths = self.depths[inds]
        
        max_d = np.max(self.depths[inds])
        dic['max_depth'] = max_d
        
        if leaf_value is not None:
            dic['values'][node] = leaf_value
        
        for i, new in enumerate(inds):
            if (left_old[new] != -1):
                dic['nodes']['left_child'][i] = inds.index(left_old[new])
            else:
                dic['nodes']['left_child'][i] = -1
                if (right_old[new] != -1):
                    dic['nodes']['right_child'][i] = inds.index(right_old[new])
                else:
                    dic['nodes']['right_child'][i] = -1

        (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
        del dic_old
        del dTree.tree_

        dTree.tree_ = Tree(n_f, n_c, n_o)
        dTree.tree_.__setstate__(dic)
        
        dTree.tree_.max_depth = max_d

        return inds.index(node)
    
    def _cut_left_right(self,node,lr):
        # Changer array parents + rules + depths
        #dic = dTree.tree_.__getstate__().copy()
        #node_to_rem = list()
        #size_init = dTree.tree_.node_count
        
        #p, b = find_parent(dTree, node)
        
        if lr == 1:
            repl_node = dTree.tree_.children_left[node]
            #node_to_rem = [node, dTree.tree_.children_right[node]]
        elif lr == -1:
            repl_node = dTree.tree_.children_right[node]
            #node_to_rem = [node, dTree.tree_.children_left[node]]
        
        repl_node = self._cut_leaf(repl_node)
        node = self.parents[repl_node]

        dic = dTree.tree_.__getstate__().copy()
        size_init = dTree.tree_.node_count
        node_to_rem = [node,repl_node]
        p, b = self.parents[node],self.bool_parents_lr[node]
        
        inds = list(set(np.arange(size_init)) - set(node_to_rem))

        dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
        dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

        if b == 1:
            dic['nodes']['right_child'][p] = repl_node
        elif b == -1:
            dic['nodes']['left_child'][p] = repl_node

        dic_old = dic.copy()
        left_old = dic_old['nodes']['left_child']
        right_old = dic_old['nodes']['right_child']
        dic['nodes'] = dic['nodes'][inds]
        dic['values'] = dic['values'][inds]

        self.parents = self.parents[inds]
        self.bool_parents_lr = self.bool_parents_lr[inds]
        self.rules = self.rules[inds]
        self.depths = self.depths[inds]
        
        max_d = np.max(self.depths[inds])
        dic['max_depth'] = max_d
        
        for i, new in enumerate(inds):
            if (left_old[new] != -1):
                dic['nodes']['left_child'][i] = inds.index(left_old[new])
            else:
                dic['nodes']['left_child'][i] = -1
            if (right_old[new] != -1):
                dic['nodes']['right_child'][i] = inds.index(right_old[new])
            else:
                dic['nodes']['right_child'][i] = -1

        (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
        del dTree.tree_
        del dic_old
    
        dTree.tree_ = Tree(n_f, n_c, n_o)
        dTree.tree_.__setstate__(dic)
        #depths = depth_array(dTree, np.linspace(0, dTree.tree_.node_count - 1, dTree.tree_.node_count).astype(int))

        dTree.tree_.max_depth = max_d
                                            
        return inds.index(repl_node)

    def _extend(self,node,subtree):
        # Changer array parents + rules + depths
        """adding tree tree2 to leaf f of tree tree1"""
        
        tree1 = self.estimator.tree_
        tree2 = subtree.tree_
        size_init = tree1.node_count
        
        dic = tree1.__getstate__().copy()
        dic2 = tree2.__getstate__().copy()
        
        size_init = tree1.node_count
        
        if depth_vtree(tree1, node) + dic2['max_depth'] > dic['max_depth']:
            #dic['max_depth'] = depth_vtree(tree1, f) + tree2.max_depth
            dic['max_depth'] = self.depths[node] + tree2.max_depth
        
        dic['capacity'] = tree1.capacity + tree2.capacity - 1
        dic['node_count'] = tree1.node_count + tree2.node_count - 1
        
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

        (Tree, (n_f, n_c, n_o), b) = tree1.__reduce__()

        tree1 = Tree(n_f, n_c, n_o)
        tree1.__setstate__(dic)
        del dic2
        del tree2
        
        """ """

        try:
            self.estimator.tree_.value[size_init:, :, subtree.classes_.astype(int)] = subtree.tree_.value[1:, :, :]
        except IndexError as e:
            print("IndexError : size init : ", size_init,
                  "\ndTree2.classes_ : ", subtree.classes_)
            print(e)

        self._compute_params(self,node=node)
        self.estimator.max_depth = self.estimator.tree_.max_depth

        return node
            
    ### @@@ ###
    
    ###########

    def UpdateSplit(self,node,feature,threshold):
        return _update_split(self,node,feature,threshold)
    
    def UpdateValue(self,node,values):
        Tree_ = self.estimator.tree_
        Tree_.value[node] = values
        Tree_.impurity[node] = ut.GINI(values)
        Tree_.n_node_samples[node] = np.sum(values)
        Tree_.weighted_n_node_samples[node] = np.sum(values)
        return node
    
    def Prune(self,node,include_node=False,lr=0,leaf_value=None):
    
        if include_node:
            """ exception si lr=0"""
            n = self._cut_left_right(node,lr)
        else:
            n = self._cut_leaf(node,leaf_value=leaf_value)

        return n

    def Extend(self,node,subtree):
        return self._extend(node,subtree)
    
    """
    def _tree_operator(self,node,mode=0,**args):
        #fusion_DT, add_to_parents, ForceCoherence,cut_,build_tree, etc...
        
        Tree_ = self.estimator.tree_

        if mode == 0 :
            #Change_node or  Update_value, test sur les **args
            if ... :
                Change_node()
            elif ... :
                Update_value
            else:

        elif mode == 1 :
            extend(Tree_,node,subtree)
        elif mode == -1 :
            prune(Tree_,node,leaf_value)
    """

    ### @@@ ###

    ###########


    def _relab(self, X, y):
        Tree_ = self.estimator.tree_
        return self

    def _ser(self, X, y, node=0):
        
        Tree_ = self.estimator.tree_

        source_values = Tree_.value[node].copy()
        node_source_label = np.argmax(source_values)
            
            #if cl_no_red is None:
            #old_size_cl_no_red = 0
            #else:
            #old_size_cl_no_red = np.sum(dTree.tree_.value[node][:, cl_no_red])
            
        # Situation où il y a des restrictions sur plusieurs classes ?
        if no_red_on_cl is not None or no_ext_on_cl is not None :
            if no_ext_on_cl:
                cl = cl_no_ext[0]
            if no_red_on_cl:
                cl = cl_no_red[0]

        if leaf_loss_quantify and ((no_red_on_cl  or  no_ext_on_cl) and maj_class == cl) and  Tree_.feature[node] == -2 :
            
            ps_rf = Tree_.value[node,0,:]/sum(Tree_.value[node,0,:])
            p1_in_l = Tree_.value[node,0,cl]/root_source_values[cl]
            
            cond_homog_unreached = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
            cond_homog_min_label = np.argmax(np.multiply(coeffs,ps_rf)) == cl
            
        ### VALUES UPDATE ###
        val = np.zeros((self.estimator.n_outputs_, self.estimator.n_classes_))

        for i in range(self.estimator.n_classes_):
            val[:, i] = list(y_target_node).count(i)
        
        #UpdateValues:
        Tree_.value[node] = val
        Tree_.n_node_samples[node] = np.sum(val)
        Tree_.weighted_n_node_samples[node] = np.sum(val)
        
        if Tree_.feature[node]== -2:
            # Extension phase :
            if original_ser:
                if y_target_node.size > 0 and len(set(list(y_target_node))) > 1:
                    
                    if max_depth is not None:
                        d = ut.depth(self.estimator,node)
                        DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                        
                        else:
                            DT_to_add = DecisionTreeClassifier()
                        
                        try:
                            DT_to_add.min_impurity_decrease = 0
                        except:
                            DT_to_add.min_impurity_split = 0
                            
                            DT_to_add.fit(X_target_node, y_target_node)
                            self.Extend(node, DT_to_add) """ extend """
                    
                return node,False
        
            else:
                bool_no_red = False
                cond_extension = False
                    
                if y_target_node.size > 0:
                    
                    if not no_ext_on_cl:
                        if max_depth is not None:
                            d = depth(self.estimator,node)
                            DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                        else:
                            DT_to_add = DecisionTreeClassifier()
            
                        try:
                            DT_to_add.min_impurity_decrease = 0
                        except:
                            DT_to_add.min_impurity_split = 0

                        DT_to_add.fit(X_target_node, y_target_node)
                        self.Extend(node, DT_to_add) """ extend """
                        #fusionDecisionTree(self.estimator, node, DT_to_add)
                    
                    else:
                        cond_maj = (maj_class not in cl_no_ext)
                        cond_sub_target = ext_cond and (maj_class in y_target_node) and (maj_class in cl_no_ext)
                        cond_leaf_loss = leaf_loss_quantify and maj_class==cl and not (cond1 and cond2)
                    
                        cond_extension = cond_maj or cond_sub_target or cond_leaf_loss
                        
                        if cond_extension:
                            if max_depth is not None:
                                d = depth(self.estimator,node)
                                DT_to_add = DecisionTreeClassifier(max_depth = max_depth - d + 1)
                            else:
                                DT_to_add = DecisionTreeClassifier()

                            try:
                                DT_to_add.min_impurity_decrease = 0
                            except:
                                DT_to_add.min_impurity_split = 0

                            DT_to_add.fit(X_target_node, y_target_node)
                            self.Extend(node, DT_to_add) """ extend """
                            #fusionDecisionTree(self.estimator, node, DT_to_add)
                        
                        else:
                            ## Compliqué de ne pas induire d'incohérence au niveau des values
                            ## en laissant intactes les feuilles de cette manière...
                            
                            Tree_.value[node] = old_values
                            Tree_.n_node_samples[node] = np.sum(old_values)
                            Tree_.weighted_n_node_samples[node] = np.sum(old_values)
                            add_to_parents(self.estimator, node, old_values) """ update values """
                            if no_red_on_cl:
                                bool_no_red = True
                                        
    
                # No red protection with values / used to flag tree parts concerned by pruning restrictions
                if no_red_on_cl and y_target_node.size == 0 and old_size_cl_no_red > 0 and maj_class in cl_no_red:
                    
                    if leaf_loss_quantify :
                        if cond1 and cond2 :
                            Tree_.value[node] = old_values
                            Tree_.n_node_samples[node] = np.sum(old_values)
                            Tree_.weighted_n_node_samples[node] = np.sum(old_values)
                            add_to_parents(self.estimator, node, old_values) """ update values """
                            bool_no_red = True
                    else:
                        Tree_.value[node] = old_values
                        Tree_.n_node_samples[node] = np.sum(old_values)
                        Tree_.weighted_n_node_samples[node] = np.sum(old_values)
                        add_to_parents(self.estimator, node, old_values) """ update values """
                        bool_no_red = True

                return node,bool_no_red
        
        """ From here it cannot be a leaf """
        ### Left / right target computation ###
        bool_test = X_target_node[:, Tree_.feature[node]] <= Tree_.threshold[node]
        not_bool_test = X_target_node[:, Tree_.feature[node]] > Tree_.threshold[node]

        ind_left = np.where(bool_test)[0]
        ind_right = np.where(not_bool_test)[0]

        X_target_node_left = X_target_node[ind_left]
        y_target_node_left = y_target_node[ind_left]

        X_target_node_right = X_target_node[ind_right]
        y_target_node_right = y_target_node[ind_right]

        if original_ser:
            new_node_left,bool_no_red_l = _ser(self,Tree_.children_left[node], X_target_node_left, y_target_node_left,
                                            original_ser = True, max_depth=max_depth)
            node, b = self.parents[new_node_left], self.bool_parents_lr[new_node_left]
            #node, b = find_parent(self.estimator, new_node_left)
                
            new_node_right,bool_no_red_r = _ser(self,Tree_.children_right[node], X_target_node_right, y_target_node_right,
                                               original_ser = True, max_depth=max_depth)
            node, b = self.parents[new_node_right], self.bool_parents_lr[new_node_right]
            #node, b = find_parent(self.estimator, new_node_right)
                                                
        else:
            new_node_left,bool_no_red_l = _ser(self,Tree_.children_left[node], X_target_node_left, y_target_node_left,original_ser=False,
                                          no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                                          no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                                          leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin,max_depth=max_depth)

            node, b = self.parents[new_node_left], self.bool_parents_lr[new_node_left]
            #node, b = find_parent(self.estimator, new_node_left)

            new_node_right,bool_no_red_r = _ser(self,Tree_.children_right[node], X_target_node_right, y_target_node_right,original_ser=False,
                                             no_red_on_cl=no_red_on_cl, cl_no_red=cl_no_red,
                                             no_ext_on_cl=no_ext_on_cl, cl_no_ext=cl_no_ext, leaf_loss_quantify=leaf_loss_quantify,
                                             leaf_loss_threshold=leaf_loss_threshold, coeffs=coeffs,root_source_values=root_source_values,Nkmin=Nkmin,max_depth=max_depth)

            node, b = self.parents[new_node_right], self.bool_parents_lr[new_node_right]
            #node, b = find_parent(self.estimator, new_node_right)

        if original_ser:
            bool_no_red = False
        else:
            bool_no_red = bool_no_red_l or bool_no_red_r

        le = ut.leaf_error(Tree_, node)
        e = ut.error(Tree_, node)

        if le <= e:
            if original_ser:
                new_node_leaf = self.Prune(node,include_node=False) """ pruning """
                #new_node_leaf = cut_into_leaf2(dTree, node)
                node = new_node_leaf
            else:
                if no_red_on_cl:
                    if not bool_no_red:
                        new_node_leaf = self.Prune(node,include_node=False) """ pruning """
                        #new_node_leaf = cut_into_leaf2(dTree, node)
                        node = new_node_leaf
            
                else:
                    new_node_leaf = self.Prune(node,include_node=False) """ pruning """
                    #new_node_leaf = cut_into_leaf2(dTree, node)
                    node = new_node_leaf

        if Tree_.feature[node] != -2:
            if original_ser:
                if ind_left.size == 0:
                    node = self.Prune(node,include_node=True,lr=-1) """ pruning """
                    #node = cut_from_left_right(dTree, node, -1)
                    
                if ind_right.size == 0:
                    node = self.Prune(node,include_node=True,lr=1) """ pruning """
                    #node = cut_from_left_right(dTree, node, 1)
            else:
                if no_red_on_cl:
                    if ind_left.size == 0 and np.sum(Tree_.value[Tree_.children_left[node]]) == 0:
                        node = self.Prune(node,include_node=True,lr=-1) """ pruning """
                        #node = cut_from_left_right(dTree, node, -1)
                        
                    if ind_right.size == 0 and np.sum(Tree_.value[Tree_.children_right[node]]) == 0:
                        node = self.Prune(node,include_node=True,lr=1) """ pruning """
                        #node = cut_from_left_right(dTree, node, 1)
                else:
                    if ind_left.size == 0:
                        node = self.Prune(node,include_node=True,lr=-1) """ pruning """
                        #node = cut_from_left_right(dTree, node, -1)
                    
                    if ind_right.size == 0:
                        node = self.Prune(node,include_node=True,lr=1) """ pruning """
                        #node = cut_from_left_right(dTree, node, 1)

        return node,bool_no_red
            
        # if tree is leaf
        #return self._build_tree(X, y)
        # Question, est ce que remplacer la feuille par un nouvel arbre va marcher?
        # Ce changement se fera-t-il bien dans self.estimator_.tree_ ?
        
        #tree.left_ = self._ser(tree.left_)
        #    tree.right_ = self._ser(tree.right_)
                
                # if condition...
        #        return self._prune(tree)

    def _strut(self, node, X, y):
        Tree_ = self.estimator.tree_

        feature_ = Tree_.feature[node]
        classes_ = self.estimator.classes_
        threshold_ = Tree_.threshold[node]
            
        old_threshold = threshold
        maj_class = np.argmax(Tree_.value[node, :].copy())
        
        if min_drift is None or max_drift is None:
            min_drift = np.zeros(self.estimator.n_features_)
            max_drift = np.zeros(self.estimator.n_features_)

        current_class_distribution = ut.compute_class_distribution(classes, Y_target_node)
        is_reached = (Y_target_node.size > 0)
        no_min_instance_targ = False
        
        if no_prune_on_cl:
            no_min_instance_targ = (sum(current_class_distribution[cl_no_prune]) == 0 )
            is_instance_cl_no_prune = np.sum(Tree_.value[node, :,cl_no_prune].astype(int))

        # If it is a leaf :
        if Tree_.feature[node_index] == -2:
            """ When to apply UpdateValue """
            if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) :
                
                ps_rf = Tree_.value[node,0,:]/sum(Tree_.value[node,0,:])
                p1_in_l = Tree_.value[node,0,cl_no_prune]/root_source_values[cl_no_prune]
                cond1 = np.power(1 - p1_in_l,Nkmin) > leaf_loss_threshold
                cond2 = np.argmax(np.multiply(coeffs,ps_rf)) == cl_no_prune
                
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune:
                if leaf_loss_quantify and (no_prune_on_cl and maj_class == cl_no_prune) and not(cond1 and cond2):
                    Tree_.value[node] = ut.current_class_distribution """ UpdateValue """
                    return node
                else:
                    return node
            else:
                Tree_.value[node] = ut.current_class_distribution
                return node

        # Only one class remaining in target :
        if (current_class_distribution > 0).sum() == 1:
            """ When to apply Pruning and how if not """
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune :
                bool_subleaf_noprune = True
                if leaf_loss_quantify:
                    bool_subleaf_noprune = contain_leaf_to_not_prune(self.estimator,cl=cl_no_prune,node=node,
                                                                    Nkmin=Nkmin,threshold=leaf_loss_threshold,coeffs=coeffs,
                                                                         root_source_values=root_source_values)
                
                if bool_subleaf_noprune :
                    rule = lib_tree.extract_rule(self.estimator,node)
                    if no_prune_with_translation :
                        node = lib_eq.ForceCoherence(self.estimator,rule,node=node,Translate=True,auto_drift=True)
                        return node
                    else:
                        node = lib_eq.ForceCoherence(self.estimator,rule,node=node)
                        return node
                            
                else:
                    node = self.Prune(node,include_node=False) """ pruning """
                    #node = cut_into_leaf2(self.estimator, node)
                    return node

            else:
                node = self.Prune(node,include_node=False) """ pruning """
                #node = cut_into_leaf2(self.estimator, node)
                return node

        # Node unreached by target :
        if not is_reached_update:
            """ When to apply Pruning and how if not """
            if no_min_instance_targ and no_prune_on_cl and is_instance_cl_no_prune :
                bool_subleaf_noprune = True
                    if leaf_loss_quantify:
                        bool_subleaf_noprune = contain_leaf_to_not_prune(self.estimator,cl=cl_no_prune,node=node,
                                                                         Nkmin=Nkmin,threshold=leaf_loss_threshold,coeffs=coeffs,
                                                                         root_source_values=root_source_values)
                    if bool_subleaf_noprune:
                        rule = lib_tree.extract_rule(self.estimator,node)
                        
                        if no_prune_with_translation :
                            node = lib_eq.ForceCoherence(self.estimator,rule,node=node,Translate=True,auto_drift=True)
                        else:
                            node = lib_eq.ForceCoherence(self.estimator,rule,node=node)
                    else:
                        #p,b = find_parent(self.estimator,node)
                        p,b = self.parents[node], self.bool_parents_lr[node]
                        node = self.Prune(node,include_node=True,lr=b) """ pruning """
                        #node = cut_from_left_right(self.estimator,p,b)

            else:
                #p,b = find_parent(self.estimator,node)
                #node = cut_from_left_right(self.estimator,p,b)
                p,b = self.parents[node], self.bool_parents_lr[node]
                node = self.Prune(node,include_node=True,lr=b) """ pruning """

            return node

        # Node threshold updates :
        """ UpdateValue """
        Tree_.value[node_index] = ut.current_class_distribution
        Tree_.weighted_n_node_samples[node] = Y_target_node.size
        Tree_.impurity[node] = ut.GINI(current_class_distribution)
        Tree_.n_node_samples[node] = Y_target_node.size
            
        # update threshold
        if type(threshold) is np.float64:
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
                                 phi,
                                 classes,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)

        Q_target_l, Q_target_r = ut.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t1,
                                                           classes)

        DG_t1 = ut.DG(Q_source_l.copy(),
                   Q_source_r.copy(),
                   Q_target_l,
                   Q_target_r)

        t2 = ut.threshold_selection(Q_source_parent,
                                 Q_source_r.copy(),
                                 Q_source_l.copy(),
                                 X_target_node,
                                 Y_target_node,
                                 phi,
                                 classes,
                                 use_divergence=use_divergence,
                                 measure_default_IG=measure_default_IG)

        Q_target_l, Q_target_r = ut.compute_Q_children_target(X_target_node,
                                                           Y_target_node,
                                                           phi,
                                                           t2,
                                                           classes)
            
       DG_t2 = ut.DG(Q_source_r.copy(),
                  Q_source_l.copy(),
                  Q_target_l,
                  Q_target_r)
                                                               
                                                               
       if DG_t1 >= DG_t2:
           Tree_.threshold[node] = t1
       else:
           Tree_.threshold[node] = t2
           # swap children
           old_child_r_id = Tree_.children_right[node]
           Tree_.children_right[node] = Tree_.children_left[node]
           Tree_.children_left[node] = old_child_r_id

        # For No Prune coherence
        ecart = Tree_.threshold[node] - old_threshold
        if Tree_.threshold[node] > old_threshold:
            if ecart > max_drift[Tree_.feature[node]] :
                max_drift[Tree_.feature[node]] = ecart
        if Tree_.threshold[node] < old_threshold:
            if ecart < min_drift[Tree_.feature[node]] :
                min_drift[Tree_.feature[node]] = ecart

        if Tree_.children_left[node] != -1:
            # Computing target data passing through node NOT updated
            #index_X_child_l = X_target_node_noupdate[:, phi] <= old_threshold
            #X_target_node_noupdate_l = X_target_node_noupdate[index_X_child_l, :]
            #Y_target_node_noupdate_l = Y_target_node_noupdate[index_X_child_l]
            # Computing target data passing through node updated
            threshold = Tree_.threshold[node]
            index_X_child_l = X_target_node[:, phi] <= threshold
            X_target_child_l = X_target_node[index_X_child_l, :]
            Y_target_child_l = Y_target_node[index_X_child_l]
            
            node = _strut(self,
                               Tree_.children_left[node],
                               X_target_child_l,
                               Y_target_child_l,
                               no_prune_on_cl=no_prune_on_cl,
                               cl_no_prune=cl_no_prune,
                               adapt_prop=adapt_prop,
                               coeffs=coeffs,
                               use_divergence = use_divergence,
                               measure_default_IG=measure_default_IG,min_drift=min_drift.copy(),max_drift=max_drift.copy(),
                               no_prune_with_translation=no_prune_with_translation,
                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,
                               root_source_values=root_source_values,Nkmin=Nkmin)
        
            node,b = self.parents[node], self.bool_parents_lr[node]
            #node,b = find_parent(self.estimator, node)

        if Tree_.children_right[node] != -1:
            # Computing target data passing through node NOT updated
            #index_X_child_r = X_target_node_noupdate[:, phi] > old_threshold
            #X_target_node_noupdate_r = X_target_node_noupdate[index_X_child_r, :]
            #Y_target_node_noupdate_r = Y_target_node_noupdate[index_X_child_r]
            # Computing target data passing through node updated
            threshold = Tree_.threshold[node]
            index_X_child_r = X_target_node[:, phi] > threshold
            X_target_child_r = X_target_node[index_X_child_r, :]
            Y_target_child_r = Y_target_node[index_X_child_r]
            
            node = _strut(self,
                               Tree_.children_right[node],
                               X_target_child_r,
                               Y_target_child_r,
                               no_prune_on_cl=no_prune_on_cl,
                               cl_no_prune=cl_no_prune,
                               adapt_prop=adapt_prop,
                               coeffs=coeffs,
                               use_divergence=use_divergence,
                               measure_default_IG=measure_default_IG,min_drift=min_drift.copy(),max_drift=max_drift.copy(),
                               no_prune_with_translation=no_prune_with_translation,
                               leaf_loss_quantify=leaf_loss_quantify,leaf_loss_threshold=leaf_loss_threshold,
                               root_source_values=root_source_values,Nkmin=Nkmin)

            node,b = self.parents[node], self.bool_parents_lr[node]
            #node,b = find_parent(self.estimator, node)

        return node

#return self




