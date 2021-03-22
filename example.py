# Get Data
from pydataset import data
dataset=data("mpg")
y_col=["cty"]
x_col=['displ', 'class' , 'cyl','manufacturer']
categoricals=['class', 'cyl','manufacturer']
Y=dataset[y_col]
X=dataset[x_col]


# Initialize model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rfcc import cluster_model
model=cluster_model(model=RandomForestRegressor,max_clusters=20,random_state=5 )


# Fit model
model.fit(X,Y,categoricals,encode_y=False, linkage_method="complete")

# Check score
print(model.score(X,Y))


# Get clusters
clusters=model.cluster_descriptions(variables_to_consider=['class','manufacturer'], continuous_measures="mean")
clusters=clusters.sort_values(by="cty-mean")
print(clusters.head(5))


# Path analysis
paths=model.path_analysis(0)
print(paths.head(0))


# Get outliers
clusters=model.cluster_descriptions(continuous_measures="mean")
clusters=clusters.sort_values(by="Nr_Obs")
outliers=clusters.head(2)
print(outliers)

# Get outlier observations
ids=model.get_observations(cluster_id=16)
print(dataset.iloc[ids,0:6])
