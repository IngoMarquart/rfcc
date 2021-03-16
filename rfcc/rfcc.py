# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:54:35 2021

@author: Ingo Marquart, ingo.marquart@esmt.org
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.spatial.distance import squareform, pdist
from rfcc.data_ops import ordinal_encode
import pandas as pd
import numpy as np
from typing import Union, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from rfcc.path_ops import recurse_path,encode_path_ordinal
from itertools import combinations
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import skew

class rfcc():

    def __init__(self,*args, model=RandomForestClassifier,  max_clusters: Optional[int] = None, logger=None, **kwargs):

        # Try to construct model
        try:
            self.model = model(*args, max_leaf_nodes=max_clusters, **kwargs)
        except:
            raise AttributeError(
                "Could not initialize {} model with parameters: {}, {}".format(model, args, kwargs))

        self.encoding_dict = None
        self.X_col_names=None
        self.y_col_names=None
        self.y_encoding_dict = None
        self.y_enc = None
        self.X_enc=None
        self.fitted = False
        self.categoricals=None
        self.cluster_list=None
        self.partition=None
        self.unique_cluster=None
        self.cluster_desc=None
        self.leaves=None
    def clusters(self):
        pass
    def cluster_descriptions(self, variables_to_consider: Optional[list] = None, continuous_measures: Optional[list]=None):
        
        assert self.fitted is True, "Model needs to be fitted to return cluster descriptions!"

        #Unpack
        stats=self.cluster_desc['stats']
        cont=self.cluster_desc['cont']
        catg=self.cluster_desc['cat']
        y=self.cluster_desc['y']
        
        medoids=[]
        for cl in self.unique_cluster:
            cl_dict={}
            cl_dict['Cluster_ID']=cl
            cl_dict['Nr_Obs']=stats[cl]['Nr']
            # y-var
            for col in y:
                cl_y=y[col][cl]
                if isinstance(cl_y,str): # categorical
                    cl_dict[col]=cl_y
                else: #Measures
                    for cm in cl_y:
                        if continuous_measures is not None:
                            if cm in continuous_measures:
                                label='{}-{}'.format(col,cm)
                                cl_dict[label]=cl_y[cm]
                        else:
                            label='{}-{}'.format(col,cm)
                            cl_dict[label]=cl_y[cm]
                        
            # cat_var
            
            for col in catg:
                cl_dict[col]=catg[col][cl]
            # cont-var
            for col in cont:
                cl_cont=cont[col][cl]
                for cm in cl_cont:
                    if continuous_measures is not None:
                        if cm in continuous_measures:
                            label='{}-{}'.format(col,cm)
                            cl_dict[label]=cl_cont[cm]
                    else:
                        label='{}-{}'.format(col,cm)
                        cl_dict[label]=cl_cont[cm]
            medoids.append(cl_dict)
        return pd.DataFrame(medoids)
            
        
        
  
    def path_analysis(self, estimator_id:Optional[int]=None):
        
        assert self.fitted is True, "Model needs to be fitted to return paths descriptions!"
        
        if estimator_id is not None:
            assert estimator_id in range(0,len(self.model.estimators_)), "No estimator for this id found."
        else:
            # TODO: Pick best tree here
            estimator_id=0
        estimator=self.model.estimators_[estimator_id]
        cluster_assignments=self.leaves[:,estimator_id]
        leaf_nodes,nr_obs=np.unique(cluster_assignments, return_counts=True)
        # Name of cluster will be included in a dict where
        # descriptions[cluster_id] gives a list of strings describing the cluster.
        descriptions=dict()
        # Loop along the clusters and extract decision paths
        # From these, we construct the name of the cluster
        for i,leaf_id in enumerate(leaf_nodes):
            # Lists to fill for this iteration
            leaf_path=list()
            leaf_feature=list()
            leaf_threshold=list()
            leaf_direction=list()
            # Start with leaf, which are the cluster nodes
            current_node=leaf_id
            # Extract paths, features and thresholds
            leaf_path,leaf_feature,leaf_threshold,leaf_direction=recurse_path(current_node,leaf_path,leaf_feature,leaf_threshold,leaf_direction,estimator)
            
            desc_path=encode_path_ordinal(self.X_col_names,leaf_path,leaf_feature,leaf_threshold,leaf_direction, self.encoding_dict)
            
            # Add output predictions for leafs
            outputs=estimator.tree_.value[leaf_id]
            output_cols=[]
            for j,ycol in enumerate(self.y_col_names):
                nameing="Output_{}".format(ycol)
                output_cols.append(nameing)
                desc_path[nameing]=outputs[j]
            
            desc_path['Nr_Obs']=nr_obs[i]
            desc_path['Cluster_ID']=leaf_id
            descriptions[leaf_id]=desc_path
            
            

        # Get number of nodes
        df=pd.DataFrame(descriptions).T
        df=df.fillna("-")
        a=['Cluster_ID','Nr_Obs']
        a.extend(output_cols)
        a.extend(self.X_col_names)
        df=df[np.intersect1d(df.columns,a)]
        return df,descriptions

    def measures(self):
        pass

    def summary(self):
        pass

    # sklearn interface
    def predict(self, X, **kwargs):

        return self.model.predict(X, **kwargs)

    def score(self, X, y, **kwargs):

        return self.model.score(X, y, **kwargs)

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], categoricals: Optional[list] = None, encode_y: Optional[bool] = False, clustering_type: Optional[str] = "consensus", t_param:Optional[float]=None, linkage_method: Optional[str] = 'average', **kwargs):

        assert isinstance(X, pd.DataFrame), "Please provide X as pd.DataFrame"
        assert isinstance(y, (pd.Series, pd.DataFrame)
                          ), "Please provide y as pd.DataFrame or pd.Series"

        if isinstance(y, pd.Series):
            y = y.to_frame()

        # Set np seed if given
        random_state=kwargs.get('random_state',0)
        np.random.seed(random_state)

        if categoricals is None:
            categoricals = X.columns
        else:
            assert isinstance(
                categoricals, list), "Please provide categorical variables as list"

        self.categoricals=categoricals
        self.X_col_names=X.columns
        self.y_col_names=y.columns
        # Prepare data by ordinal encoding
        X, self.encoding_dict, self.X_enc = ordinal_encode(X, categoricals, return_enc=True)
        if encode_y:
            y, self.y_encoding_dict, self.y_enc = ordinal_encode(
                y, y.columns, return_enc=True)

        n = y.shape[0]
        y_index = np.array(y.index)
        # Need a slightly different call for 1 dimensional outputs
        if y.shape[1] <= 1:
            self.model.fit(X, np.ravel(y), **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

        # Model has been fitted
        self.fitted = True

        # Get nr_obs and save the leaf nodes
        nr_obs = X.shape[0]
        self.leaves = self.model.apply(X)
        nr_leave_nodes = [len(np.unique(self.leaves[:, i]))
                          for i in range(0, self.leaves.shape[1])]
        avg_nr_leaves = int(np.mean(nr_leave_nodes))
        # Regular clustering, or consensus clustering
        if clustering_type == "consensus":
            # Create one distance matrix for all estimators
            # TODO: use lil sparse matrix and a cutoff to populate matrices
            # if nr_obs is very high!
            distance_matrix = np.zeros([nr_obs, nr_obs], dtype=float)
            nr_estimator = len(self.model.estimators_)
            # Get distances in each decision tree
            for estimator in self.model.estimators_:
                # Extract the cluster labels
                obs_cluster_labels = estimator.apply(X)
                leaf_nodes = np.unique(obs_cluster_labels)
                leaf_path_dict = {}
                # Get paths for each decision tree
                for leaf_id in leaf_nodes:
                    leaf_path = list()
                    leaf_feature = list()
                    leaf_threshold = list()
                    leaf_direction = list()

                    leaf_path, leaf_feature, leaf_threshold, leaf_direction = recurse_path(
                        leaf_id, leaf_path, leaf_feature, leaf_threshold, leaf_direction, estimator)
                    leaf_path.insert(0, leaf_id)
                    leaf_path_dict[leaf_id] = [
                        leaf_path, leaf_feature, leaf_threshold, leaf_direction]
                # Now create a distance matrix for each leaf node based
                # on path distances
                # TODO: Condition this on threshold and feature distances as well
                leaf_distance_matrix = np.zeros(
                    [len(leaf_nodes), len(leaf_nodes)])
                leaf_distance_matrix = pd.DataFrame(
                    leaf_distance_matrix, index=leaf_nodes, columns=leaf_nodes)
                for dyad in combinations(leaf_nodes, 2):
                    p1 = leaf_path_dict[dyad[0]][0]
                    p2 = leaf_path_dict[dyad[1]][0]
                    path_length = len(np.setxor1d(p1, p2))
                    leaf_distance_matrix.loc[dyad[0],
                                             dyad[1]] = path_length
                leaf_distance_matrix += leaf_distance_matrix.T
                # Normalize path distance
                leaf_distance_matrix=leaf_distance_matrix/np.max(leaf_distance_matrix.values)
                # Build the rows for each leaf->obs, and then apply
                # to observation the fitting leaf node
                for leaf_id in leaf_nodes:
                    row = np.array(obs_cluster_labels, dtype=float)
                    row = np.zeros(obs_cluster_labels.shape, dtype=float)
                    for alter in leaf_nodes:
                        row[obs_cluster_labels == alter] = leaf_distance_matrix.loc[leaf_id, alter]
                    # Add result for applicable observations to the distance matrix
                    distance_matrix[obs_cluster_labels == leaf_id, :] += row
            # Normalize
            distance_matrix=squareform(distance_matrix)/nr_estimator        
        else:
            # Regular clustering based on similarities of leaf nodes
            distance_matrix = pdist(self.leaves, metric='hamming')

        # Run Clustering
        dendogram = linkage(distance_matrix, method=linkage_method)
        if t_param == None:
            self.cluster_list = fcluster(dendogram, avg_nr_leaves, 'maxclust')
        else:
            self.cluster_list = fcluster(dendogram, t_param, 'distance')
        self.unique_cluster=np.unique(self.cluster_list)
        self.partition = {y_index[i]: self.cluster_list[i] for i in range(0, n)}
        
        # Create descriptions
        if encode_y:
            y_cols=y.columns
            y_ind=y.index
            y=pd.DataFrame(self.y_enc.inverse_transform(y), columns=y_cols, index=y_ind)
        X.loc[:,categoricals]=self.X_enc.inverse_transform(X[categoricals])
        self.__create_cluster_descriptions(X,y,encode_y,categoricals)
        
      
    def __create_cluster_descriptions(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], y_categorical:bool, categoricals: Optional[list] = None, variables_to_consider: Optional[list] = None):
        
        assert self.fitted is True, "Model needs to be fitted to create cluster descriptions!"
        
        outcome=y.columns
        rcdata=pd.merge(y, X, left_index=True, right_index=True)
        
        continuous=np.setdiff1d(X.columns,categoricals)
        descriptions={}
        
        # Continuous variables
        column_dict={}
        for col in continuous:
            cluster_dict={}
            for cl in self.unique_cluster:
                cl_mask=self.cluster_list==cl
                subset=X.loc[cl_mask,col]
                cluster_dict[cl]={'mean':np.mean(subset), 'median':np.median(subset), 'std':np.std(subset),'max':np.max(subset), 'min': np.min(subset),'skew':skew(subset)}
            column_dict[col]=cluster_dict
        descriptions['cont']=column_dict
        
        # Categorical variables
        column_dict={}
        for col in categoricals:
            
            cluster_dict={}
            for cl in self.unique_cluster:
                cl_mask=self.cluster_list==cl
                subset=X.loc[cl_mask,col]
                values, number=np.unique(subset,return_counts=True)
                total=np.sum(number)
                ratios=np.round(number/total,2)
                desc=['{}: {}%'.format(x,y) for x,y in zip(values,ratios)]
                desc=', '.join(desc)
                cluster_dict[cl]=desc
            column_dict[col]=cluster_dict
        descriptions['cat']=column_dict
        
        # Outcome either categorical or not
        column_dict={}
        for col in outcome:
            
            if y_categorical:
                cluster_dict={}
                for cl in self.unique_cluster:
                    cl_mask=self.cluster_list==cl
                    subset=y.loc[cl_mask,col]
                    values, number=np.unique(subset,return_counts=True)
                    total=np.sum(number)
                    ratios=np.round(number/total,2)
                    desc=['{}: {}%'.format(x,y) for x,y in zip(values,ratios)]
                    desc=', '.join(desc)
                    cluster_dict[cl]=desc
                column_dict[col]=cluster_dict    
            else:
                cluster_dict={}
                for cl in self.unique_cluster:
                    cl_mask=self.cluster_list==cl
                    subset=y.loc[cl_mask,col]
                    cluster_dict[cl]={'mean':np.mean(subset), 'median':np.median(subset), 'std':np.std(subset),'max':np.max(subset), 'min': np.min(subset),'skew':skew(subset)}
                column_dict[col]=cluster_dict                                                
        descriptions['y']=column_dict
        # Further stats
        cluster_dict={}
        for cl in self.unique_cluster:
            cl_mask=self.cluster_list==cl
            subset=X.loc[cl_mask,:]
            cluster_dict[cl]={'Nr':subset.shape[0]}
        descriptions['stats']=cluster_dict
        
        self.cluster_desc=descriptions
