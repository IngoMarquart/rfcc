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
@pytest.fixture
def get_mpg_dataset_cat():
    dataset=data("mpg")
    y_col=["cty"]
    x_col=np.setdiff1d(dataset.columns,y_col)
    categoricals=list(np.setdiff1d(x_col, ['displ','hwy', 'cyl']))
    Y=dataset[y_col]
    X=dataset[x_col]
    return Y,X,categoricals


@pytest.mark.rfcc
def test_fit(get_mpg_dataset):

    Y,X,categoricals=get_mpg_dataset_cat
    
    model=cluster_model(model=RandomForestRegressor,max_clusters=15 )
    
    model.fit(X,Y,categoricals,encode_y=False)
    df=model.cluster_descriptions(continuous_measures=['std'])
    nr_obs=np.sum(df['Nr_Obs'])
    assert int(nr_obs)==X.shape[0]


@pytest.mark.rfcc
def test_path_analysis(get_mpg_dataset):

    Y,X,categoricals=get_mpg_dataset_cat
    
    model=cluster_model(model=RandomForestRegressor,max_clusters=15 )
    
    model.fit(X,Y,categoricals,encode_y=False)
    
    leaves=model.leaves
    nr_leaves=len(leaves[0])
    
    df=model.path_analysis(0)
    
    assert nr_leaves==len(df)
    