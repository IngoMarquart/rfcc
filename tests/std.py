# -*- coding: utf-8 -*-
from sklearn.datasets import load_wine
import pandas as pd
from pydataset import data
import numpy as np
from rfcc.data_ops import ordinal_encode
from rfcc.rfcc import rfcc
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import pandas as pd
import scipy as sp
from itertools import product
from tqdm import tqdm, tnrange,trange
import tqdm.notebook as tq
def get_mpg_dataset_cat():
    dataset=data("mpg")
    y_col=["cty"]
    x_col=np.setdiff1d(dataset.columns,y_col)
    categoricals=list(np.setdiff1d(x_col, ['displ','hwy', 'cyl']))
    Y=dataset[y_col]
    X=dataset[x_col]
    return Y,X,categoricals


seeds=list(range(500,502))
Y,X,categoricals=get_mpg_dataset_cat()
df_list=[]
full_list=[]
for cluster_size in trange(2,15, desc="Depth"):
    outliers_new=[]
    outliers_old=[]
    std_new=[]
    std_old=[]
    mean_new=[]
    mean_old=[]
    cty_std_new=[]
    cty_std_old=[]
    nr_cl_new=[]
    nr_cl_old=[]
    for seed in tqdm(seeds, leave=True, position = 0, desc="Depth {}".format(cluster_size)):
        
        model=rfcc(model=RandomForestRegressor,max_clusters=cluster_size,random_state=seed )
        model.fit(X,Y,categoricals,encode_y=False, linkage_method="average", t_param=0.2)
        df=model.cluster_descriptions()
        full_list.append(df)
        outliers_new.append(len(np.where(df['Nr_Obs']==1)[0]))
        mean_new.append(np.mean(df['Nr_Obs']))
        std_new.append(np.std(df['Nr_Obs']))
        nr_cl_new.append(len(df['Nr_Obs']))
        cty_std_new.append(np.mean(df['cty-std']))
        model=rfcc(model=RandomForestRegressor,max_clusters=cluster_size,random_state=seed )
        model.fit(X,Y,categoricals,encode_y=False, clustering_type="old", linkage_method="average", t_param=0.2)
        df=model.cluster_descriptions()
        outliers_old.append(len(np.where(df['Nr_Obs']==1)[0]))
        cty_std_old.append(np.mean(df['cty-std']))
        std_old.append(np.std(df['Nr_Obs']))
        nr_cl_old.append(len(df['Nr_Obs']))
        mean_old.append(np.mean(df['Nr_Obs']))

    df_list.append({'Clustersize':cluster_size, 'Nr_RCC':np.mean(nr_cl_new),'Nr_Binary':np.mean(nr_cl_old), 'RFCC outcome std': np.mean(cty_std_new), 'Binary outcome std': np.mean(cty_std_old), 'RFCC avg':np.mean(mean_new),'Binary avg':np.mean(mean_old), 'RFCC std':np.mean(std_new),'Binary std':np.mean(std_old)})
    
df2=pd.DataFrame(df_list)
df2=df2.set_index('Clustersize')
df2[['RFCC avg','Binary avg']].plot()
df2[['RFCC std','Binary std']].plot()
df2[['RFCC outcome std','Binary outcome std']].plot()
df2['RFCC']=df2['RFCC outcome std']/df2['RFCC avg']
df2['Binary']=df2['Binary outcome std']/df2['Binary avg']
df2[['RFCC','Binary']].plot()
df2[['Nr_RCC','Nr_Binary']].plot()