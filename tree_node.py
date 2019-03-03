import numpy as np

class TreeNode(object):
    """
    The fundamental data structure for a Decision Tree.
    Root, Branches and Leaves are all TreeNode objects.
    """
    def __init__(self, features=None, parent=None, recommendation=None, split_feature=None):
        """
        Params
        ------
        features: list
            The feature values corresponding to this node : 1 or 0
            None, if the feature has not been split.
        parent: TreeNode
            None, for root node
        recommendation: float
            The recommended departure time to leave home, for days that fall in this node.
        split_feature: int
            The index of the feature on which this node's parent are split.
        """
        self.parent = parent
        self.hi_branch = None
        self.lo_branch = None
        self.is_leaf = True
        self.split_feature = split_feature
        self.features = features
        self.recommendation = recommendation

    def attempt_split(self, data, err_fn, n_min):
        """
        Takes all the datapoints that correspond to this node.
        A designated error function and the minimum number of training data points
        allowed per leaf.
        Try to split this node into two child nodes, decide whether and where to split.
        
        Parameters
        ----------
        data: DataFrame, consisting of features for each of the data points
        err_fn: Function 
            Error function to determine fitness of a split. Choose a split
            that minimises the combined error of the two resulting branches.
        n_min: int
            The minimum number of nodes in each leaf. Prevents overfitting.
            This is called the 'Stopping Criteria'
        
        Returns
        -------
        success: boolean
            If a split happen, return True. Otherwise False
        """
        success = False
        n_features = len(data.columns)
        # initialise the root node properly
        if(self.features is None):
            self.features = [None]*n_features
        node_data = self.find_members(data)

        feature_candidates = [i for i,j in enumerate(self.features) if (j is None)] #hold the indices of the features that have not been split(or, j=None)
        best_feature = -1
        best_split_score = 1e10 #this has to be minimised, this is the error
        best_hi_recommendation = self.recommendation
        best_lo_recommendation = self.recommendation

        #Check for each potential feature dimensions for split quality
        if(len(feature_candidates)>0):
            np.random.shuffle(feature_candidates)
            for i_feature in feature_candidates:
                hi_data = node_data.loc[node_data.iloc[:,i_feature]==1, :]
                lo_data = node_data.loc[node_data.iloc[:,i_feature]==0, :]

                if hi_data.shape[0] >= n_min and lo_data.shape[0]>= n_min:
                    hi_score, hi_recommendation = err_fn(list(hi_data.index))
                    lo_score, lo_recommendation = err_fn(list(lo_data.index))
                    split_score = hi_score + lo_score
                    if split_score < best_split_score:
                        best_split_score = split_score
                        best_feature = i_feature
                        best_hi_recommendation = hi_recommendation
                        best_lo_recommendation = lo_recommendation
                        success = True
                        #inherently it is a greedy strategy
        if(success):
            hi_features = list(self.features)
            hi_features[best_feature] = 1
            self.hi_branch = TreeNode(parent=self, features=hi_features,recommendation=best_hi_recommendation)
            lo_features = list(self.features)
            lo_features[best_feature] = 0
            self.lo_branch = TreeNode(parent=self, features=lo_features,recommendation=best_lo_recommendation)
            self.is_leaf = False
            self.split_feature = best_feature
        return success

    def find_members(self, data):
        """
        Find all of the dates within features that
        belong to this node.
        Params
        ------
        data: DataFrame

        Returns
        -------
        member_data: DataFrame
            A subset of the rows of the features dataframe
        """
        member_data = data
        for i_feature, feature in enumerate(self.features):
            if(self.features[i_feature] is not None):
                member_data = member_data.loc[member_data.iloc[:,i_feature] == feature]
        return member_data
