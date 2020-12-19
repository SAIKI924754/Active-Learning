# Activelearning
Getting unlabelled data has been a very simple thing these days. But, it is indeed a very difficult task to acquire labeled data.
Active learning is such a framework that comes to the rescue when you have limited data and yet you require a better model accuracy.
Querying intelligently to identify the most informative instances is the underlying principle of active learning.
Key factors in building any active learning model is the urcertainity measure we choose and the query strategy we apply.

# Query Stratergies:

  ## 1. Uncertainity Sampling
   Any active learner, when it has been presented with a set of unlabelled examples, it extracts the most useful example and it provides the same to get labelled.
   Measuring the usefulness of the prediction is the first calculated for each example and a decision is made depending on the usefulness.
   Classification Uncertainity, classification margin and classification entropy are the three built in measures availale in the modAL documentation for Active Learning 
   Pool Based sampling and Stream based sampling are the different ways in which instances can be sent to query for measuring usefulness.
   
   ## Pool based sampling 
   The below example presents the application of an Active learner onto the fetch_covtype dataset using pool-based sampling. 
    Here a large set of unlabbeled data and a set of labelled data with very very few number of examples compared to that in the unlabelled data set is taken. 
    The principle of pool ased sampling can be explained as :
    "Queries are selectively drawn from the pool, which is usually assumed to be closed (i.e., static or non-changing), although this is not strictly necessary. Typically, instances are queried in a greedy fashion, according to an informativeness measure used to evaluate all instances in the pool (or, perhaps if U is very large, some subsample thereof)".
    
   Along with our pool-based sampling strategy, modAL’s modular design allows you to vary parameters surrounding the active learning process, including the core estimator and query strategy. In this example, we use scikit-learn’s k-nearest neighbors classifier as our estimator and default to modAL’s uncertainty sampling query strategy.

```ruby
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
import numpy as np

# Set our RNG seed for reproducibility.

RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
```
```ruby
X_raw = data['data']
y_raw = data['target']

print(X_raw.shape)
print(X_raw)
print(y_raw.shape)
print(y_raw)
```
```ruby
from sklearn.decomposition import PCA

# Define our PCA transformer and fit it onto our raw dataset.
pca = PCA(n_components=2, random_state=RANDOM_STATE_SEED)
transformed_fetch_covtype = pca.fit_transform(X=X_raw)

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

#Isolate the data we'll need for plotting.
x_component, y_component = transformed_fetch_covtype[:, 0], transformed_fetch_covtype[:, 1]

Plot our dimensionality-reduced (via PCA) dataset.
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(x=x_component, y=y_component, c=y_raw, cmap='viridis', s=50, alpha=8/10)
plt.title('forest cover type classes after PCA transformation')
plt.show()
```
```ruby

# Isolate our examples for our labeled dataset.
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples - 1, size=481012)
print(training_indices.shape)

X_test = X_raw[training_indices]
y_test = y_raw[training_indices]

X_raw = np.delete(X_raw, training_indices, axis=0)
y_raw = np.delete(y_raw, training_indices, axis=0)
print(X_raw.shape)
print(y_raw.shape)
```
```ruby
n_labeled_examples = X_raw.shape[0]
training_indices = np.random.randint(low=0, high=n_labeled_examples - 1, size=10000)

X_train = X_raw[training_indices]
y_train = y_raw[training_indices]

# Isolate the non-training examples we'll be querying.
X_pool = np.delete(X_raw, training_indices, axis=0)
y_pool = np.delete(y_raw, training_indices, axis=0)
print(X_pool.shape)
print(y_pool.shape)
```

```ruby
from sklearn.neighbors import KNeighborsClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

# Specify our core estimator along with it's active learning model.
knn = KNeighborsClassifier(n_neighbors=3)
learner = ActiveLearner(estimator=knn, query_strategy=uncertainity_sampling, X_training=X_train, y_training=y_train,)
 ```
```ruby 
#Isolate the data we'll need for plotting.
predictions = learner.predict(X_raw)
is_correct = (predictions == y_raw)


unqueried_score = learner.score(X_raw, y_raw)
print(unqueried_score)
```
```ruby
#Plot our classification results.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.scatter(x=x_component[is_correct],  y=y_component[is_correct],  c='g', marker='+', label='Correct',   alpha=8/10)
ax.scatter(x=x_component[~is_correct], y=y_component[~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
ax.legend(loc='lower right')
ax.set_title("ActiveLearner class predictions (Accuracy: {score:.3f})".format(score=unqueried_score))
plt.show()

#change this variable to change the number of additional data points, 20 is 10%
N_QUERIES = 5


performance_history = [unqueried_score]
```
```ruby
rawpool = X_pool
tarpool = y_pool

# Allow our model to query our unlabeled dataset for the most
# informative points according to our query strategy (uncertainty sampling).

for index in range(N_QUERIES):
        
    query_index, query_instance = learner.query(rawpool)

# Teach our ActiveLearner model the record it has requested.
    X, y = rawpool[query_index].reshape(1, -1), tarpool[query_index].reshape(1, )
    learner.teach(X=X, y=y)

# Remove the queried instance from the unlabeled pool.
    rawpool, tarpool = np.delete(rawpool, query_index, axis=0), np.delete(tarpool, query_index)

# Calculate and report our model's accuracy.
    model_accuracy = learner.score(rawpool, tarpool)
        
    print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))

# Save our model's performance for plotting.
    performance_history.append(model_accuracy)
```
```ruby

# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)
ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
ax.set_ylim(bottom=0, top=1)
ax.grid(True)
ax.set_title('Active learning accuracy')
ax.set_xlabel('Number of queries')
ax.set_ylabel('Classification Accuracy')
plt.show()
```
  ## Stream based Sampling
  In addition to pool-based sampling, the stream-based scenario can also be implemented easily with modAL. In this case, the labels are not queried from a pool of instances. Rather, they are given one-by-one for the learner, which queries for its label if it finds the example useful. For instance, an example can be marked as useful if the prediction is uncertain, because acquiring its label would remove this uncertainty.
  Along with our stream-based sampling strategy, modAL’s modular design allows you to vary parameters surrounding the active learning process, including the core estimator and query strategy. In this example, we use scikit-learn’s RandomForest classifier as our estimator and default to modAL’s uncertainty sampling query strategy.

```ruby
  import numpy as np
# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 123
np.random.seed(RANDOM_STATE_SEED)
```
```ruby
from sklearn.datasets import fetch_covtype

data = fetch_covtype()
X_full = data['data']
y_full = data['target']

print(X_full.shape[1])
print(X_full)
print(y_full.shape[0])
print(y_full)
```
```ruby
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
   plt.imshow(im)
    plt.title('The shape to learn')
    plt.show()
```
```ruby
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
# assembling initial training set
n_initial = 5
initial_idx = np.random.choice(range(len(X_full)), size=n_initial, replace=False)
X_train, y_train = X_full[initial_idx], y_full[initial_idx]
```
```ruby
# initialize the learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_train, y_training=y_train)
unqueried_score = learner.score(X_full, y_full)

print('Initial prediction accuracy: %f' % unqueried_score)
```

``` ruby  
from modAL.uncertainty import classifier_uncertainty

performance_history = [unqueried_score]

# learning until the accuracy reaches a given threshold
while learner.score(X_full, y_full) < 0.9:
    stream_idx = np.random.choice(range(len(X_full)))
    if classifier_uncertainty(learner, X_full[stream_idx].reshape(1, -1)) >= 0.4:
        learner.teach(X_full[stream_idx].reshape(1, -1), y_full[stream_idx].reshape(-1, ))
        new_score = learner.score(X_full, y_full)
        performance_history.append(new_score)
        print('sample no. %d queried, new accuracy: %f' % (stream_idx, new_score))
 ```
 ```ruby
 # Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')
```

  ## 2. Query by Committee
  The strategy of uncertainity sampling doesn' work in all the cases. Query by Committee is yet another useful technique which comes handy in the cases where uncertaininty sampling doesn't prove effective. Bias towards the actual learner resulting in missing the important examples is one of the drawbacks of uncertainity sampling.  keeQuery by Committee fixes this by keeping several hypotheses at the same time, selecting queries where disagreement occurs between them. In this example, we shall see how this works in the case of the fetch_covtype dataset.
  
```ruby
import numpy as np
# Set our RNG seed for reproducibility.
RANDOM_STATE_SEED = 1
np.random.seed(RANDOM_STATE_SEED)
```
```ruby
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_covtype
data= fetch_covtype()
```
```ruby
# visualizing the classes
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(data['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=data['target'], cmap='viridis', s=50)
    plt.title('The fetch_covtype dataset')
    plt.show()
``` 
```ruy
from copy import deepcopy
# generate the pool
X_pool = deepcopy(data['data'])
y_pool = deepcopy(data['target'])
```
```ruby
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
# initializing Committee members
n_members = 2
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 2
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)
```
```ruby
# initializing learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)
```
```ruby
# assembling the committee
committee = Committee(learner_list=learner_list)

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(data['data']), cmap='viridis', s=50)
        plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
    plt.show()
```
```ruby
unqueried_score = committee.score(data['data'], data['target'])

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(data['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee initial predictions, accuracy = %1.3f' % unqueried_score)
    plt.show()
    
performance_history = [unqueried_score]
```
```ruby
# query by committee
n_queries = 20
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    performance_history.append(committee.score(data['data'], data['target']))
 # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
```
```ruby
# visualizing the final predictions per learner
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(data['data']), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
    plt.show()
 ```
 ```ruby
#visualizing the Committee's predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(data['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee predictions after %d queries, accuracy = %1.3f'
              % (n_queries, committee.score(data['data'], data['target'])))
    plt.show()
```
```ruby
# Plot our performance over time.
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

plt.show()
```

 

  


  
