#!/usr/bin/env python
# coding: utf-8

# # Lecture 1 - Jet tagging with neural networks

# > A first look at training deep neural networks to classify jets in proton-proton collisions.

# ## Learning objectives
# 
# * Understand what jet tagging is and how to frame it as a machine learning task
# * Understand the main steps needed to train and evaluate a jet tagger
# * Learn how to download and process data with the 🤗 Datasets library
# * Gain an introduction to the fastai library and how to push models to the Hugging Face Hub

# ## References
# 
# * Chapter 1 of [_Deep Learning for Coders with fastai & PyTorch_](https://github.com/fastai/fastbook) by Jeremy Howard and Sylvain Gugger.
# * [_The Machine Learning Landscape of Top Taggers_](https://arxiv.org/abs/1902.09914) by G. Kasieczka et al.
# * [_How Much Information is in a Jet?_](https://arxiv.org/abs/1704.08249) by K. Datta and A. Larkowski.

# ## The task and data

# For the first few lectures, we'll be analysing the [_Top Quark Tagging_ dataset](https://huggingface.co/datasets/dl4phys/top_tagging), which is a famous benchmark that's used to compare the performance of jet classification algorithms. The dataset consists of around 2 million Monte Carlo simulated events in proton-proton collisions that have been clustered into jets.
# 
# Framed as a supervised machine learning task, the goal is to train a model that can classify each jet as either a top-quark signal or quark-gluon background.

# ```{figure} ./images/jet-tagging.png
# ---
# scale: 100%
# name: jet-tagging
# ---
# Figure reference: [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772)
# ```

# ## Setup

# In[ ]:


# Uncomment and run this cell if using Colab, Kaggle etc
# %pip install fastai==2.6.0 datasets git+https://github.com/huggingface/huggingface_hub


# In[1]:


# Check we have the correct fastai version
import fastai

assert fastai.__version__ == "2.6.0"


# ## Import libraries

# In[3]:


from datasets import load_dataset
from fastai.tabular.all import *
from huggingface_hub import from_pretrained_fastai, notebook_login, push_to_hub_fastai
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


# In[4]:


import datasets

# Suppress logs
datasets.logging.set_verbosity_error()


# ## Getting the data

# We will use the [🤗 Datasets](https://github.com/huggingface/datasets) library to download and process the datasets that we'll encounter in this course. 🤗 Datasets provides smart caching and allows you to process larger-than-RAM datasets by exploiting a technique called _memory-mapping_ that provides a mapping between RAM and filesystem storage.
# 
# To download the Top Quark Tagging dataset from the [Hugging Face Hub](https://huggingface.co/datasets/dl4phys/top_tagging), we can use the `load_dataset()` function:

# In[8]:


top_tagging_ds = load_dataset("dl4phys/top_tagging")


# If we look inside our `top_tagging_ds` object

# In[9]:


top_tagging_ds


# we see it is similar to a Python dictionary, with each key corresponding to a different split. And we can use the usual dictionary syntax to access an individual split:

# In[10]:


top_tagging_ds["train"]


# The `Dataset` object is one of the core data structures in 🤗 Datasets and behaves like an ordinary Python `list`, so we can query its length:

# In[11]:


len(top_tagging_ds["train"])


# or access a single element by its index:

# In[12]:


top_tagging_ds["train"][0]


# Here we see that a single row is repesented as a dictionary where the keys correspond to the column names. Since we won't need the top-quark 4-vector columns, let's remove them along with the `ttv` one:

# In[13]:


top_tagging_ds = top_tagging_ds.remove_columns(
    ["truthE", "truthPX", "truthPY", "truthPZ", "ttv"]
)


# Although 🤗 Datasets provides a lot of low-level functionality for preprocessing datasets, it is often conventient to convert a `Dataset` object to a Pandas `DataFrame`. To enable the conversion, 🤗 Datasets provides a `set_format()` method that allows us to change the output format of the dataset: 

# In[14]:


# Convert output format to DataFrames
top_tagging_ds.set_format("pandas")
# Create DataFrames for the training and test splits
train_df, test_df = top_tagging_ds["train"][:], top_tagging_ds["test"][:]
# Peek at first few rows
train_df.head()


# As we can see, each row consists of 4-vectors $(E_i, p_{x_i}, p_{y_i}, p_{z_i})$ that correspond to the maximum 200 particles that make up each jet. We can also see that each jet has been padded with zeros, since most won't have 200 particles. We also have an `is_signal_new` column that indicates whether the jet is a top quark signal (1) or QCD background (0).
# 
# Now that we've had a look at a sample of the raw data, let's take a look at how we can convert it to a format that is suitable for neural networks!

# ## Introducing fastai
# 
# To train our model, we'll use the [fastai library](https://github.com/fastai/fastai). fastai is the most popular framework for training deep neural networks with PyTorch and provides various application-specific classes for different types of deep learning data structures and architectures. It is also designed with a _layered API_, which means:
# 
# * We can use high-level components to quickly and easily get state-of-the-art results in standard deep learning domains
# * Low-level components can be mixed and matched to build new approaches
# 
# In particular, this approach will allow us in later lessons to use pure PyTorch code to define our models, and then let fastai take care of the training loop (which is often an error-prone process).
# 
# **Basics of the API**
# 
# The most common steps one takes when training a model in fastai are:
# 
# * Create `DataLoaders` to feed batches of data to the model
# * Create a `Learner` which wraps the architecture, optimizer, and data, and automatically chooses an appropriate loss function where possible
# * Find a good learning rate
# * Train your model
# * Evaluate your model
# 
# Let's go through each of these steps to build a neural network that can classify top quark jets from the QCD background!

# ### From data to DataLoaders

# To wrangle our data in a format that's suitable for training neural nets, we need to create an object called `DataLoaders`. To turn our dataset into a `DataLoaders` object we need to specify:
# 
# * What type of data we are dealing with (tabular, images, etc)
# * How to get the examples
# * How to label each example
# * How to create the validation set
# 
# fastai provides a number of classes for different kinds of deep learning datasets and problems. In our case, the data is in a _tabular_ format (i.e. a table of rows and columns), so we can use the `TabularDataLoaders` class:

# In[15]:


# Downsample to ~0.5 if you're running on Colab / Kaggle which have limited RAM
frac_of_samples = 1.0
train_df = train_df.sample(int(frac_of_samples * len(train_df)), random_state=42)

features = list(train_df.drop(columns=["is_signal_new"]).columns)
splits = RandomSplitter(valid_pct=0.20, seed=42)(range_of(train_df))

dls = TabularDataLoaders.from_df(
    df=train_df,
    cont_names=features,
    y_names="is_signal_new",
    y_block=CategoryBlock,
    splits=splits,
    bs=1024,
)


# Let's unpack this code a bit. The first thing we've specified is which columns of our dataset correspond to _continuous features_ via the `cont_names` argument. To do this, we've simply grabbed all column names from our `DataFrame`, except for the label column `is_signal_new`. Next, we've specified which column is the target in `y_names` and that this is a _categorical feature_ with `CategoryBlock`. Finally we've specified the training and validation splits with `RandomSplitter` and picked a batch size of 1,024 examples.
# 
# After we've defined a `DataLoaders` object, we can take a look at the data by using the `show_batch()` method:

# In[16]:


dls.show_batch()


# This looks like the format we want: we have a matrix of numerical features encoded in the 4-vectors, along with a target denoted by the `is_signal_new` column.

# ### Create a Learner

# We can now create the `Learner` for this data. fastai provides various application-specific learner classes, each of which come with a set of good defaults for training. In our case, we'll use the `tabular_learner` class:

# In[17]:


learn = tabular_learner(
    dls, layers=[200, 200, 50, 50], metrics=[accuracy, RocAucBinary()]
)


# By default, `tabular_learner` creates a neural network with two hidden layers and 200 and 100 activations each. This works great for small datasets, but since our dataset is quite large, we've increased the depth of the network by adding two more layers. This also matches the architecture chosen in Section 3.2.2 of [_The Machine Learning Landscape of Top Taggers_](https://arxiv.org/abs/1902.09914) review that we'll compare to later.
# 
# We've also provided two common classification metrics to track during training: accuracy and the Area Under the ROC Curve (ROC AUC). We'll look at ROC AUC in more detail later, so for now let's take a look at our network with the `summary()` method:

# In[18]:


learn.summary()


# Here we can see that this particular network has around 215,000 parameters - although this sounds like a lot, it's actually a very small model by modern standards (e.g. in natural language processing, some models have hundreds of billions of parameters!).

# ### Find a good learning rate

# The learning rate is one of the most important hyperparameters involved in training neural networks, so it's important to make sure you've picked a good one. We'll see in the next lesson exactly how this parameter impacts training, but for now it is enough to know that:
# 
# * If our learning rate is too low, it will take a long time to train the model and there is a good chance of overfitting.
# * If our learning rate is too high, the training process can diverge.
# 
# To handle these two extremes, fastai provides a _learning rate finder_ that tracks the loss as we increase the learning rate. You can see this in action by using the `lr_find()` method of any `Learner`:

# In[19]:


learn.lr_find()


# From this curve we can see that the loss hits a minimum around a learning rate of $3 \times 10^{-1}$, so we should select a learning rate lower than this point. The `lr_find()` method provides a handy heuristic to pick the learning rate 1-2 orders of magnitude less than the minimum, as indicated by the orange dot. 

# ### Train your model

# In the above learning rate plot, it appears a learning rate of around $10^{-3}$ would be good, so let's choose that and train our models for 3 epochs:

# In[20]:


learn.fit_one_cycle(n_epoch=3, lr_max=1e-3)


# Once the model is trained, we can view the results in various ways. A simple approach is to use the `show_results()` method to compare the model errors:

# In[21]:


learn.show_results()


# Here we can see that model made a handful of errors, which is expected since our accuracy is only around 82%. However, evaluating our model’s predictions on the same data it was trained on is almost always a recipe for disaster! Why? The problem is that the model may memorise the structure of the data it sees and fail to provide good predictions when shown new data. Let's see how we can evaluate our model on examples from the _test set_ that it has never seen.

# ### Evaluate your model

# The learners in fastai are equipped with `predict()` and `get_preds()` methods that allow one to evaluae the model on new data. To use them, we'll need a new `DataLoader` which we can create by simply passing in a `DataFrame` of the test events:

# In[22]:


test_dl = learn.dls.test_dl(test_items=test_df)


# Now that we have a `DataLoader`, it's a simple matter to compute the predictions with the `get_preds()` method:

# In[23]:


preds, targs = learn.get_preds(dl=test_dl)


# Let's take a look at the first few values of `preds` and `targs`:

# In[24]:


preds[:5], targs[:5]


# Here we can see that they are _tensors_. In PyTorch, a tensor is similar to the arrays that you may be familiar with in Numpy. Tensors have a rank that can be inspected by using the `size()` method:

# In[25]:


preds.size(), targs.size()


# In this case, we see that `preds` is a rank-2 tensor (i.e. a matrix), while `targs` is rank-1 (a vector). Note that each dimension in `preds` corresponds to the _probabilities_ of the model for the two classes (signal vs background). We can visualise the distribution of the probabilities for each class by filtering the `preds` tensor according to the ground truch labels and then plotting the result as a histogram:

# In[26]:


signal_test = preds[:, 1][targs.flatten() == 1].numpy()
background_test = preds[:, 1][targs.flatten() == 0].numpy()

plt.hist(signal_test, histtype="step", bins=20, range=(0, 1), label="Signal")
plt.hist(background_test, histtype="step", bins=20, range=(0, 1), label="Background")
plt.xlabel("Probability")
plt.ylabel("Events/bin")
plt.yscale("log")
plt.xlim(0, 1)
plt.legend(loc="lower right", frameon=False)
plt.show()


# We see that although the model assigns high (low) probabilities to the signal (background) events, a fair amount of the signal events overlap with background ones.  To handle this, one usually defines a "cut" or threshold that only includes events above that value. For example, if define the cut at 0, then all the events are counted and the signal efficiency $\epsilon_S$ and background efficiency $\epsilon_B$ are both 1. As we increase the cut, we reject more and more background events and the result is a curve with $\epsilon_{B,S}$ ranging from 0 to 1. 
# 
# This curve is equivalent to the Reciever Operating Characteristic (ROC) curve which plots the true positive rate
# 
# $$ \mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}} \,, \qquad \mathrm{TP\, (FP)} = \mathrm{number\, of\, true\, (false) \,positives}\,, $$
# 
# against the false positive rate FPR, where the FPR is the ratio of negative instances that are incorrectly classified as positive. In general there is a tradeoff between these two quantities: the higher the TPR, the more false positives (FPR) the classifier produces.
# 
# To visualise the ROC curve for our model's predictions, we can use the handy `roc_curve()` function from scikit-learn:

# In[27]:


# fpr = epsilon_B, tpr = epsilon_S
fpr, tpr, thresholds = roc_curve(y_true=targs, y_score=preds[:, 1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], ls="--", color="k")
plt.xlabel(r"$\epsilon_B$")
plt.ylabel(r"$\epsilon_S$")
plt.tight_layout()


# A perfect classifier would have a ROC curve with all signal and background events correctly identified, i.e. an Area Under the Curve (AUC) of 1. Let's compute this area along with the accuracy on the test set:

# In[28]:


acc_test = accuracy_score(targs, preds.argmax(dim=-1))
auc_test = auc(fpr, tpr)
print(f"Accuracy: {acc_test:.4f}")
print(f"AUC: {auc_test:.4f}")


# Since the AUC is dominated by values at large $\epsilon_B$, it is common to also report the background rejection at a fixed signal efficiency (often 30%). We can do that by defining an interpolating function across the `tpr` and `fpr` values as follows:

# In[29]:


background_eff = interp1d(tpr, fpr)
background_eff_at_30 = background_eff(0.3)
print(f"Backround rejection at signal efficiency 0.3: {1/background_eff_at_30:0.3f}")


# Comparing these results again the [_The Machine Learning Landscape of Top Taggers_](https://arxiv.org/abs/1902.09914) review, shows that our baseline model falls short of the models in the review, which get a typical accuracy of 93% and an AUC of 98%. 

# ```{figure} ./images/top-tagging-scores.png
# ---
# scale: 25%
# name: top-landscape
# ---
# Figure reference: [The Machine Learning Landscape of Top Taggers](https://arxiv.org/abs/1902.09914)
# ```

# Let's see if we can train a better model by choosing a clever representation of the input data! Before doing that, let's collect this evaluation logic in a function that we can reuse later:

# In[30]:


def compute_metrics(learn, test_df):
    test_dl = learn.dls.test_dl(test_items=test_df)
    preds, targs = learn.get_preds(dl=test_dl)
    fpr, tpr, _ = roc_curve(y_true=targs, y_score=preds[:, 1])
    acc_test = accuracy_score(targs, preds.argmax(dim=-1))
    auc_test = auc(fpr, tpr)
    background_eff = interp1d(tpr, fpr)
    background_eff_at_30 = background_eff(0.3)

    print(f"Accuracy: {acc_test:.4f}")
    print(f"AUC: {auc_test:.4f}")
    print(
        f"Backround rejection at signal efficiency 0.3: {1/background_eff_at_30:0.3f}"
    )
    return fpr, tpr


# In[31]:


fpr_baseline, tpr_baseline = compute_metrics(learn, test_df)


# ## Jet representations

# In any machine learning problem, how we represent the data often has a large impact on the performance of the models we train. For jet tagging, the most common approaches one finds in the literature include:
# 
# * **Jets as images.** A jet image is a pixelated grayscale image, where the pixel intensity represents the energy (or transverse momentum) of all particles that deposited energy in a particular location in the $\eta-\phi$ plane. Typically, _convolutional neural networks (CNNs)_ are used to process the images and we'll ecplore these architectures in a future lesson.
# * **Jets as sequences.** Here the idea is to order the particles in a jet (usually by $p_T$) and use sequence-based architectures like _recurrent neural networks (RNNs)._
# * **Jets as graphs.** This approach treats each jet as a generic graph of nodes and edges. Graph neural networks (which we'll also encounter later in the course) excel on this tpe of data.
# * **Jets as sets.** A generalisation of the previous representations, this approach simply treats each jets as an unordered collection or point cloud of 4-vectors.
# * **Theory-inspired representations.** Instead of representing the jets in formats to match specific neural network architectures, these approaches use results on IR safety from QCD to represent the jets as a simplified set of features. Fully-connected neural networks are then trained on these features.
# 
# You can find more details about each representation in a nice [review article](https://arxiv.org/abs/1709.04464) from 2017. 
# 
# ```{figure} ./images/jets.png
# ---
# scale: 40%
# name: jet-substructure
# ---
# Figure reference: [Jet Substructure at the Large Hadron Collider](https://arxiv.org/abs/1709.04464)
# ```
# 
# In this lesson and the next, we'll use one of the theory-inspired representations called $N$-subjettiness. Let's take a look.

# ### Representing jets with $N$-subjettiness observables

# $N$-subjettiness observables quantify how much of the radiation in a jet is aligned along different subjet axes. Although originally used for analytic approaches to distinguish different decays and event topologies, these observable can also be used as inputs for machine learning models and provide strong discriminating power. 
# 
# To be precise, an $N$-subjettiness observable $\tau_N^{(\beta)}$ measures the radiation about $N$ axes in the jet according to an angular exponent $\beta>0$:
# 
# $$ \tau_N^{(\beta)} = \frac{1}{p_{T,J}} \sum_{i\in J} p_{T,i} \min \left\{ R_{1,i}^\beta, R_{1,i}^\beta, \ldots , R_{1,i}^\beta \right\} $$
# 
# Here $p_{T,J}$ is the transverse momentum of the jet, $p_{T,i}$ is the transverse momentum of particle $i$ in the jet, and $R_{K,i}$ is the distance in the $\eta-\phi$ plane of particle $i$ to axis $K$.
# 
# To measure substructure in a jet, one thus needs to measure a suitable number of $N$-subjettiness observables. In practice this is done by specifying the corrdinates of $M$-body phase space in terms of $3M - 4$ $N$-subjettiness observables:
# 
# $$ \left\{ \tau_1^{(0.5)}, \tau_1^{(1)}, \tau_1^{(2)}, \tau_2^{(0.5)}, \tau_2^{(1)}, \tau_2^{(2)}, \ldots , \tau_{m-1}^{(0.5)}, \tau_{m-1}^{(1)}, \tau_{m-1}^{(2)} \right\} $$
# 
# To see how we can use this basis as features for a neural network, we have computed $N$-subjettiness observables up through 6-body phase space using the [pyjet library](https://github.com/scikit-hep/pyjet). You can download these features via the `load_dataset()` function as follows:

# In[32]:


nsubjet_ds = load_dataset("dl4phys/top_tagging_nsubjettiness")


# As before, we'll convert our `Dataset` object to a pandas `DataFrame`:

# In[33]:


nsubjet_ds.set_format("pandas")
train_df, test_df = nsubjet_ds["train"][:], nsubjet_ds["test"][:]
train_df.head()


# Following Section 3.2.2 of [_The Machine Learning Landscape of Top Taggers_](https://arxiv.org/abs/1902.09914) revie, we've also included the jet mass and jet $p_T$ as input variables to allow the network to learn physical scales. 
# 
# Let's now train a model using these features. As before, we need to first define our `DataLoaders` object:

# In[34]:


features = list(train_df.drop(columns=["label"]).columns)
splits = RandomSplitter(valid_pct=0.20, seed=42)(range_of(train_df))

dls = TabularDataLoaders.from_df(
    df=train_df,
    cont_names=features,
    y_names="label",
    y_block=CategoryBlock,
    splits=splits,
    bs=1024,
)


# And just like before it's a good idea to sanity check your data is formatted correctly with the `show_batch()` method:

# In[35]:


dls.show_batch()


# This looks good, so the last step is to create a `Learner` and find a good learning rate:

# In[36]:


learn = tabular_learner(
    dls, layers=[200, 200, 50, 50], metrics=[accuracy, RocAucBinary()]
)

learn.lr_find()


# This curve is similar to what we found before so let's pick a learning rate of $10^{-3}$ and train for 3 epochs:

# In[37]:


learn.fit_one_cycle(n_epoch=3, lr_max=1e-3)


# We can already see that training on the $N$-subjettiness features has produced a better model than our baseline, which achieved around 83% and $91%$ accuracy and AUC score respectively. Let's wrap up by computing these metrics on the test set with our `compute_metrics()` function:

# In[38]:


test_df = nsubjet_ds["test"].to_pandas()
fpr_nsubjet, tpr_nsubjet = compute_metrics(learn, test_df)


# This is much better and now just a 1-2% the classifiers reported in the review paper! We can also compare both models by plotting the background rejection rate against the signal efficiency:

# In[39]:


fig, ax = plt.subplots()

plt.plot(tpr_baseline, 1 / (fpr_baseline + 1e-6), label="Baseline")
plt.plot(tpr_nsubjet, 1 / (fpr_nsubjet + 1e-6), label="6-body N-subjettiness")
plt.xlabel("Signal befficiency $\epsilon_{S}$")
plt.ylabel("Background rejection $1/\epsilon_{B}$")
plt.xlim(0, 1)
plt.yscale("log")
plt.legend()
plt.show()


# ## Saving and sharing the model
# 
# We've seen in this lecture how to load and prepare datasets for deep neural nets, and how to train the models with fastai. But what happens if you want to save your model for future use, or to simply reproduce the results from your paper in PRL 🙃? 
# 
# One way to do this is by use the `save()` method of the `Learner`, which will store your model in a format called [pickle](https://docs.python.org/3/library/pickle.html). This is great if you're doing quick experimentation, but at some point you might want to share the model with a colleague or the wider research community. 
# 
# In the same way we were able to download a dataset from the Hugging Face Hub, it is also possible to share fastai models on the platform! To do so, you'll first need to:
# 
# * Create a Hugging Face account (it's free): https://huggingface.co/join
# * [Optional] Join the [Deep Learning for Particle Physicists](https://huggingface.co/dl4phys) organisation to share your models with the rest of the class.
# 
# Once you're created an account, you can log into the Hub with the following helper function:

# In[40]:


notebook_login()


# This will display a widget in which you can enter a Hugging Face token - you can find details on how to create tokens in the [Hub documentation](https://huggingface.co/docs/hub/security#user-access-tokens). Once you're logged in, pushing our model to the Hub is simple via the `push_to_hub_fastai()` function:

# In[46]:


user_or_org = "dl4phys"  # This can also be your Hub username, e.g. lewtun
model_id = "lewtun-top-tagging-nsubs"  # Change this to something different
repo_id = f"{user_or_org}/{model_id}"

push_to_hub_fastai(
    learner=learn,
    repo_id=repo_id,
    commit_message="Add new Learner",
)


# This will push the model's weights to the Hub, as well as a `pyproject.toml` file that defines the environment in which the `Learner` was created. Now that we've pushed our model to the Hub, we can now download it on any machine where fastai is installed! This is really handy when you need to quickly reproduce the results from your paper, e.g. here is how we use the `from_pretrained_fastai()` function to download the model and re-compute our scores on the test set:

# In[42]:


learner = from_pretrained_fastai(repo_id)
_, _ = compute_metrics(learner, test_df)


# ## Exercises
# 
# * Try changing the network architecture by adjusting the `layers` argument in `tabular_learner()` on the $N$-subjettiness features. What happens if you keep the number of nodes fixed and increase the number of layers? Similarly, what happens if you have just a single layer, but increase the number of nodes on that layer?
# * Push one of your new models to the `dl4phys` organisation on the Hugging Face Hub.
# * Train a model on the $N$-subjettiness features without including the jet mass and $p_T$. Does this have a positive or negative impact on performance?
# * Train a model using 2-body and 4-body $N$-subjettiness features to see if the performance is saturated with a smaller number of features.

# In[ ]:




