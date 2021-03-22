# -*- coding: utf-8 -*-
from sklearn.datasets import load_wine
import pandas as pd
from pydataset import data
import numpy as np
from rfcc.data_ops import ordinal_encode
from rfcc import cluster_model
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

def get_mpg_dataset_cat():
    dataset=data("mpg")
    y_col=["cty"]
    x_col=np.setdiff1d(dataset.columns,y_col)
    categoricals=list(np.setdiff1d(x_col, ['displ','hwy', 'cyl']))
    Y=dataset[y_col]
    X=dataset[x_col]
    return Y,X,categoricals


seeds=list(range(500,1000))
cluster_sizes=list(range(2,40))
Y,X,categoricals=get_mpg_dataset_cat()
df_list=[]
for cluster_size in cluster_sizes:
    outliers_new=[]
    outliers_old=[]
    for seed in tqdm(seeds, leave=True, position = 0, desc="Depth {}".format(cluster_size)):
        
        model=cluster_model(model=RandomForestRegressor,max_clusters=cluster_size,random_state=seed )
        model.fit(X,Y,categoricals,encode_y=False,t_param=None)
        df=model.cluster_descriptions(continuous_measures=['mean'])
        outliers_new.append(len(np.where(df['Nr_Obs']==1)[0]))
        
        model=cluster_model(model=RandomForestRegressor,max_clusters=cluster_size,random_state=seed )
        model.fit(X,Y,categoricals,encode_y=False, clustering_type="old",t_param=None)
        df=model.cluster_descriptions(continuous_measures=['mean'])
        outliers_old.append(len(np.where(df['Nr_Obs']==1)[0]))
    df_list.append({'Clustersize':cluster_size, 'Outliers_New':np.mean(outliers_new),'Outliers_Old':np.mean(outliers_old)})
    
df=pd.DataFrame(df_list)
df=df.set_index('Clustersize')
df.plot()