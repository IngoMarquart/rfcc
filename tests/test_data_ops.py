from sklearn.datasets import load_wine
import pandas as pd
from pydataset import data
import numpy as np
from rfcc.data_ops import ordinal_encode
import pytest

@pytest.fixture
def get_wine_dataset():
    
    dataset=load_wine()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    Y = pd.DataFrame(dataset.target)
    categoricals=[]
    return Y,X,categoricals
    

@pytest.fixture
def get_mpg_dataset():
    dataset=data("mpg")
    y_col=["cty","hwy"]
    x_col=np.setdiff1d(dataset.columns,y_col)
    categoricals=list(np.setdiff1d(x_col, ['year', 'displ', 'cyl']))
    Y=dataset[y_col]
    X=dataset[x_col]
    return Y,X,categoricals
    

@pytest.mark.data_operations
def test_encoding(get_mpg_dataset):
    
    Y,X,categoricals=get_mpg_dataset
    
    X_cat,encoding_dict, enc= ordinal_encode(X,categoricals,return_enc=True)
    X_cat2=enc.inverse_transform(X_cat[categoricals])
    
    assert all(X[categoricals]==X_cat2)

