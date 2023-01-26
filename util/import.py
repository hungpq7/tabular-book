# general python
import os
import glob
import logging
import warnings
import datetime as dt
import dateutil.relativedelta as du
import re
from sspipe import p, px

# general data science
import numpy as np
import pandas as pd
import janitor
import scipy import stats
import numpy.random as rd
import sympy as sym

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# general machine learning
from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from feature_engine.transformation import BoxCoxTransformer, YeoJohnsonTransformer

# machine learning algorithms
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# time series analysis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt

# time series forecast
from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter, CutoffSplitter
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV, temporal_train_test_split
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.base import ForecastingHorizon as FH
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.compose import MultiplexForecaster
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError as MAPE
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError as MASE
from sktime.pipeline import make_pipeline
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Detrender
from sktime.transformations.series.detrend import Deseasonalizer
from sktime.transformations.series.difference import Differencer
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.naive import NaiveVariance

# text mining
import nltk
import spacy

# config
np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_colwidth = 1000
pd.options.display.max_columns = 500
pd.options.display.max_rows = 1000
plt.style.use(['seaborn', 'seaborn-whitegrid'])
