
# Get Data
from pydataset import data
dataset=data("mpg")
y_col=["cty"]
x_col=['displ', 'class' , 'cyl','manufacturer']
encode=['class', 'cyl','displ']
Y=dataset[y_col]
X=dataset[x_col]


# Initialize model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rfcc.model import cluster_model
model=cluster_model(model=RandomForestRegressor,max_clusters=20,random_state=5 )


# Fit model
model.fit(X,Y,encode,encode_y=False, linkage_method="complete")

# Check score
print(model.score(X,Y))

print(model.predict(X))

# Get clusters
clusters=model.cluster_descriptions(variables_to_consider=['class','manufacturer'], continuous_measures="mean")
clusters=clusters.sort_values(by="cty-mean")
print(clusters.head(5))


# Path analysis
paths=model.path_analysis(0)
print(paths.head(1))


# Get outliers
clusters=model.cluster_descriptions(continuous_measures="mean")
clusters=clusters.sort_values(by="Nr_Obs")
outliers=clusters.head(2)
print(outliers)

# Get outlier observations
ids=model.get_observations(cluster_id=16)
print(dataset.iloc[ids,0:6])

## Repeat the above with a classification analysis

# Get Data
from pydataset import data
dataset=data("mpg")
y_col=["manufacturer"]
x_col=['displ', 'class' , 'cyl','cty']
Y=dataset[y_col]
X=dataset[x_col]

# Initialize model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rfcc.model import cluster_model
model=cluster_model(model=RandomForestClassifier,random_state=5 )

# Fit model
model.fit(X,Y, t_param=0.4, linkage_method="complete")
# Check score
print(model.score(X,Y))


# Get clusters
clusters=model.cluster_descriptions(continuous_measures="mean")
print(clusters.head(5))


# Path analysis
paths=model.path_analysis(0)
print(paths.head(1))


