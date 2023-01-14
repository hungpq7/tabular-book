#!/usr/bin/env python
# coding: utf-8

# # 1. Basic concepts
# [Hypothesis testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) is a method of statistical inference that tests the validity of a claim about the population, using sample data. It makes use of the following concepts.

# ### Hypotheses
# - The [null hypothesis](https://en.wikipedia.org/wiki/Null_hypothesis) (denoted $H_0$): a common view whose validity needs to be tested.
# - The [alternative hypothesis](https://en.wikipedia.org/wiki/Alternative_hypothesis) (denoted $H_1$): what will be believed if $H_0$ is rejected.

# ### Significance level
# [Significance level](https://en.wikipedia.org/wiki/Statistical_significance) (denoted $\alpha$) is a pre-selected number ranges from $0$ to $1$, indicates the probability of rejecting the null hypothesis. Common values of $\alpha$ is $0.05$ and $0.01$. A related concept to significance level is [confidence level](https://en.wikipedia.org/wiki/Confidence_interval) (denoted $\gamma=1-\alpha$). Each significance level corresponds to a critical value ($c$).

# ### Test statistic
# Being one of the most important factors, [test statistic](https://en.wikipedia.org/wiki/Test_statistic) (denoted $T$) is the transformed data that follows a theoretical distribution. Since the probability distribution function is known, it allows calculating the probability value, telling which hypothesis is more likely to happen. Each test statistic is represented by a fraction where:
# - The numerator is the power of signal
# - The denominator is the power of noise

# ### p-value
# [p-value](https://en.wikipedia.org/wiki/P-value) is the probability of making [type I error](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) - rejecting $H_0$ when it's true. It represents the probability of $H_0$ being true, and when this probability is less than $\alpha$, $H_0$ should be rejected. The smaller the p-value is, the stronger the evidence that $H_0$ should be rejected.
# - A p-value less than $0.05$ indicates the difference is significant, meaning there is a probability of less than $5\%$ that the null hypothesis is correct. Therefore, $H_0$ is rejected and $H_1$ is accepted.
# - A p-value higher than $0.05$ indicates the difference is not significant. In this case, $H_1$ is rejected but $H_0$ is failed to be rejected.

# In[1]:


from scipy import stats
import numpy as np
mu, std = 0, 1
dist = stats.norm(mu, std)


# In[2]:


# compute critical value (c) given a significance level (alpha)
# 2-tailed test
alpha = 0.05
crit = dist.isf(alpha/2)
crit


# In[3]:


# compute p-value (p) given a test statistic (t)
# 2-tailed test
test = 1.96
test = np.abs(test)
pval = dist.cdf(-test) + dist.sf(test)
pval


# ### Descriptive statistics
# For populations:
# - $N$: population size
# - $\mu$: population mean
# - $\sigma$: population standard deviation
# - $\sigma^2$: population variance
# - $p$: proportion of successes in population
# 
# For samples:
# - $n$: sample size
# - $\hat\mu$ or $\bar x$: sample mean
# - $\hat\sigma$ or $\mbox{SD}$: sample standard deviation
# - $\hat\sigma^2$ or $s^2$: sample variance
# - $\hat p$: proportion of successes in sample
# - $\mbox{SE}_{\mu}$: standard error of mean
# - $\mbox{SE}_p$: standard error of proportion

# # Hypothesis testing summary
# Type  |Usage| Test statistic             
# :----------|:--------------|:--------------------------
# Z-test|1. Comparing the means of 1 or 2 populations <br> 2. Comparing the proportions of 1 or 2 populations|$Z$|
# F-test|Comparing the variances of 2 populations|$F$|
# t-test|Comparing the means of 1 or 2 populations|$T$|
# Chi-squared test|1. Comparing the propotions of 3 or more populations <br> 2. Testing of qualitative variables replationship|$\chi^2$|
# ANOVA|Comparing the means of 3 population or more|$F$|
# KS test|Testing of distribution|$D$|

# # 1. Z-test
# The usage of [Z-test](https://en.wikipedia.org/wiki/Z-test):
# - Comparing the mean of a population with a specific number or comparing the means of two populations
# - Comparing the proportion of a population with a specific number or comparing the proportions of two populations
# 
# Assumptions:
# - Populations are normally distributed
# - Samples are random and must have more than 30 observations
# - Population variances are already known (only in mean Z-test)

# In[4]:


import math
import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest
from scipy import stats

from IPython.display import display, Markdown


# In[5]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# In[47]:


def compute_pvalue(testStat, distribution, alternative):
    if alternative in ['two-sided', '2s', 'other than']:
        signH1 = '!='
        pValue = distribution.sf(np.abs(testStat)) * 2
    elif alternative in ['larger', 'right']:
        signH1 = '>'
        pValue = distribution.sf(testStat)
    elif alternative in ['smaller', 'left']:
        signH1 = '<'
        pValue = distribution.cdf(testStat)
    return signH1, pValue

def process_decision(pValue, alpha):
    if pValue <= alpha:
        signTest = '<'
        mess = 'reject Null Hypothesis'
    elif pValue > alpha:
        signTest = '>'
        mess = 'fail to reject Null Hypothesis'
    return signTest, mess


# In[50]:


def ZTestMean(x1, var1, x2=None, var2=None, A=0, alternative='two-sided', alpha=0.05):
    # compute x1 statistics
    x1 = np.array(x1)
    mu1, n1 = x1.mean(), x1.size
    
    # compute x2 statistics
    if x2 is not None and var2 is not None:
        x2 = np.array(x2)
        mu2, n2 = x2.mean(), x2.size
        objective = 'mean1 - mean2'
    elif x2 is None and var2 is None:
        var2 = 0
        mu2, n2 = 0, np.inf
        objective = 'mean'
    
    # compute test statistic
    se = np.sqrt(var1/n1 + var2/n2)
    testStat = (mu1-mu2-A) / se
    
    # compute p-value
    signH1, pValue = compute_pvalue(testStat, stats.norm, alternative)
    
    # make decision
    signTest, mess = process_decision(pValue, alpha)
    
    print(
        f'Alternative Hypothesis: {objective} {signH1} {A}' '\n'
        f'p-value = {pValue:.4f} {signTest} {alpha} --> {mess}'
    )


# In[51]:


def ZTestProportion(p1, n1, p2=None, n2=None, A=0, alternative='two-sided', alpha=0.05):
    # compute x2 statistics
    if p2 is not None and n2 is not None:
        objective = 'p1 - p2'
    elif p2 is None and n2 is None:
        p2, n2 = 0, np.inf
        objective = 'p'
    
    # compute test statistic
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    testStat = (p1-p2-A) / se
        
    # compute p-value
    signH1, pValue = compute_pvalue(testStat, stats.norm, alternative)
    
    # make decision
    signTest, mess = process_decision(pValue, alpha)
    
    print(
        f'Alternative Hypothesis: {objective} {signH1} {A}' '\n'
        f'p-value = {pValue:.4f} {signTest} {alpha} --> {mess}'
    )


# ## 1.1. One-sample mean
# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# Given a random sample sized $N=500$ of people's income from a population having the standard deviation $\sigma=5000$. With the significant level $\alpha=0.05$, can we conclude that the mean of the population $\mu=A=14000$?
# 
# First, state the hypotheses from the information:
# - $H_0: \mu = 14000$
# - $H_1: \mu \neq 14000$
# 
# Since it is a two-tailed test, the critical value will be $z_{\alpha/2}=z_{0.025} = 1.96$. If $|T|>1.96$, reject $H_0$ and accept $H_1$. However, in this example, $|T|=0.63$ and the corresponding p-value is $0.2643$, so $H_0$ cannot be rejected. The formula for the test statistic is:
# 
# $$T = \frac{\hat{\mu}-A}{\mbox{SE}_{\mu}}\quad\text{for } \text{SE}_\mu = \sqrt{\frac{\sigma^2}{N}}$$

# In[39]:


ZTestMean(df.age, var1=140, A=59)


# ## 1.2. Two-sample mean
# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# The average income of male is $5000$ higher than female, true or false? Given $\alpha = 0.05$, population standard deviations of income of male and female are $\sigma_1=7000$ and $\sigma_2=5000$, consecutively.
# 
# The hypotheses:
# - $H_0: \mu_1 = \mu_2+5000$
# - $H_1: \mu_1 > \mu_2+5000$
# 
# This is a right-tailed test, $z_{\alpha}=z_{0.05} = 1.64$ will be taken. If $T>1.64$, reject $H_0$ and conclude that the average income of male is higher than female. In this example, $T=2.57$ and the corresponding p-value is $0.0051$. The formula for the test statistic is:
# 
# $$T=\frac{(\hat{\mu}_1-\hat{\mu}_2)-A}{\mbox{SE}_{\mu}}
# \quad\text{for }\text{SE}_\mu=\sqrt{\frac{\sigma_1^2}{N_1}+\frac{\sigma_2^2}{N_2}}$$

# In[32]:


x1 = df.query('gender=="male"').income
x2 = df.query('gender=="female"').income

ZTestMean(x1=x1, x2=x2, var1=7000**2, var2=5000**2, A=5000, alternative='2s')


# ## 1.3. One-sample proportion
# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# In a large consignment of food packets, a random sample of $n=100$ packets revealed that 5 packets were leaking. Can we conclude that the population contains at least $A=10\%$ of leaked packets at $\alpha=0.05$?
# 
# The hypotheses:
# - $H_0: p\geq0.1$
# - $H_1: p<0.1$
# 
# This is a left-tailed test, $H_0$ will be rejected if $T<-z_{0.05}=-1.64$. For $T=-2.294$, the corresponding p-value is $0.011$ ($<0.05$). The formula for the test statistic is:
# 
# $$T = \frac{\hat{p}-A}{\mbox{SE}_p}
# \quad\text{for }\text{SE}_p=\sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$$

# In[45]:


ZTestProportion(p1=5/100, n1=100, A=0.1)


# ## 1.4. Two-sample proportion
# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# A machine turns out 16 imperfect articles in a sample of $n_1=500$. After maintaining, it turns 3 imperfect articles in a sample of $n_2=100$. Has the machine improved after maintaining at the significance level of $\alpha=0.05$?
# 
# The hypotheses:
# - $H_0: p_1=p_2$
# - $H_1: p_1>p_2$
# 
# If $T>z_{0.05}=1.64$, reject $H_0$. The formula for the test statistic is:
# 
# $$T = \frac{(\hat{p}_1-\hat{p}_2)-A}{\mbox{SE}_p}
# \quad\text{for }\text{SE}_p=\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{N_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{N_2}}$$

# In[46]:


ZTestProportion(p1=16/500, n1=500, p2=3/100, n2=100, A=0)


# # 2. F-test
# The usage of [F-test](https://en.wikipedia.org/wiki/F-test):
# - Comparing the variances of two populations
# - Being used in one-way ANOVA to compare the means between groups (section 2.4)
# - Being used in multivariate linear regression to testing the significant of R-squared (section 3.2)
# 
# Assumption:
# - Populations are normally distributed
# - The two random samples are independent

# In[10]:


import math
import numpy as np
import pandas as pd
from scipy import stats
from collections import namedtuple


# In[11]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# In[37]:


def compute_pvalue(testStat, distribution, alternative):
    if alternative in ['two-sided', '2s', 'other than']:
        signH1 = '!='
        pValue = distribution.sf(np.abs(testStat)) * 2
    elif alternative in ['larger', 'right']:
        signH1 = '>'
        pValue = distribution.sf(testStat)
    elif alternative in ['smaller', 'left']:
        signH1 = '<'
        pValue = distribution.cdf(testStat)
    return signH1, pValue

def process_decision(pValue, alpha):
    if pValue <= alpha:
        signTest = '<'
        mess = 'reject Null Hypothesis'
    elif pValue > alpha:
        signTest = '>'
        mess = 'fail to reject Null Hypothesis'
    return signTest, mess


# In[65]:


def FTest(x1, x2=None, A=0, alternative='two-sided', alpha=0.05):
    objective = 'var1 / var2'
    
    # compute x1 statistics
    x1 = np.array(x1)
    var1, dof1 = x1.var(), x1.size-1
    
    # compute x2 statistics
    x2 = np.array(x2)
    var2, dof2 = x2.var(), x2.size-1
    
    # compute test statistic
    testStat = var1 / var2 / A
    
    # compute p-value
    signH1, pValue = compute_pvalue(testStat, stats.f(dof1, dof2), alternative)
    
    # make decision
    signTest, mess = process_decision(pValue, alpha)
    
    print(
        f'Alternative Hypothesis: var1 {signH1} {A}*var2' '\n'
        f'p-value = {pValue:.4f} {signTest} {alpha} --> {mess}'
    )


# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# With the significance level $\alpha=0.05$, compare the population variances of income of male and female.
# 
# The hypotheses:
# - $H_0: \sigma^2_1 = 5\sigma^2_2$
# - $H_1: \sigma^2_1 > 5\sigma^2_2$
# 
# If p-value $<0.05$: reject $H_0$. The formula for the test statistic is:
# 
# $$T = \frac{1}{A}\frac{\hat{\sigma}_1^2}{\hat{\sigma}_2^2}$$

# In[53]:


x1 = df[df['gender']=='male'].income
x2 = df[df['gender']=='female'].income


# In[66]:


FTest(x1, x2, A=5, alternative='larger')


# # 3. t-test
# The [t-test](https://en.wikipedia.org/wiki/Student%27s_t-test) is any statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis.

# In[15]:


import math
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from collections import namedtuple


# In[16]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# ## 3.1. One-sample t-test
# The usage of [one-sample t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test): To compare the mean of a population with a number, when the population variance is unknown.
# 
# Assumption:
# - The population is normally distributed
# - The sample is random

# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# With the confidence level of $\alpha=0.05$, the mean of income is $13000$ or not?
# 
# The hypotheses:
# - $H_0: \mu=13000$
# - $H_1: \mu\neq13000$
# 
# The formula for the test statistic is:
# 
# $$T = \frac{\hat{\mu}-A}{\hat\sigma/\sqrt{n}}$$

# In[17]:


# new in scipy version 1.6.0
stats.ttest_1samp(df.income, 13000, alternative='two-sided')


# In[18]:


pg.ttest(x = df.income, y = 13000.0, tail='two-sided')


# ## 3.2. Independent two-sample
# The usage of [independent two_sample t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test): to compare the means of two populations using their independent samples. A F-test should be used first to check the equality of the two population variances.
# 
# Assumptions:
# - Two populations are normally distributed
# - Two samples are independent and random
# - Two variances are equal

# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# With $\alpha=0.05$, the average income of male and female are equal, true or false?
# 
# The hypotheses:
# - $H_0: \mu_1 = \mu_2$
# - $H_1: \mu_1 \neq \mu_2$
# 
# If $\sigma_1^2 \neq \sigma_2^2$ (this example - already tested in section 2.2), the formula for the test statistic is:
# 
# $$T = \frac{\hat{\mu}_1-\hat{\mu}_2-A}{\sqrt{\frac{\hat{\sigma}_1^2}{n_1}+\frac{\hat{\sigma}_2^2}{n_2}}}$$
# 
# If $\sigma_1^2 = \sigma_2^2$, the test statistic is:
# 
# $$T = \frac{\hat{\mu}_1-\hat{\mu}_2-A}{\hat\sigma_p \sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}$$
# 
# where
# 
# $$\hat\sigma_p = \sqrt{\frac{(n_1-1)\hat{\sigma}_1^2 + (n_2-1)\hat{\sigma}_2^2}{n_1+n_2-2}}$$
# 
# is the pooled standard deviation of the two samples.

# In[19]:


x1 = df[df['gender']=='male'].income
x2 = df[df['gender']=='female'].income


# In[20]:


stats.ttest_ind(x1, x2, equal_var=False, alternative='two-sided')


# In[21]:


pg.ttest(x=x1, y=x2, correction=True, tail='two-sided')


# ## 3.3. Dependent paired samples t-test
# The usage of [dependent paired samples t-test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples): to compare two population means, given their dependent samples. A paired samples t-test calculates the diffrence between paired observation and then performs a one-sample t-test.
# 
# Assumptions:
# - The two populations should be both normally distributed
# - The two random samples come in pairs (before and after data for example)
# - Same sample sizes

# In[22]:


x1 = [72,77,84,79,74,67,74,77,79,89]
x2 = [65,68,77,73,66,61,66,71,71,78]


# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# With $\alpha=0.05$, the average weight after is 8 kg less than before, true or false?
# 
# The hypotheses:
# - $H_0: \mu_1-\mu_2\geq8$
# - $H_1: \mu_1-\mu_2<8$
# 
# The test statistic is:
# 
# $$T = \frac{\hat\mu_1-\hat\mu_2-A}{\hat\sigma_d/\sqrt n} = \frac{\hat\mu_d-A}{\hat\sigma_d/\sqrt n}$$
# where
# - $\hat\mu_d$ is the sample mean of the differences
# - $\hat\sigma_d$ is the sample standard deviation of the differences

# In[23]:


# using scipy and pg can not change the mu value
stats.ttest_rel(x1, x2, alternative='two-sided')


# In[24]:


pg.ttest(x=x1, y=x2, paired=True, tail='two-sided')


# In[25]:


def pair_ttest(data1, data2, mu=0, alternative='two-sided'):
    x1 = np.array(data1)
    x2 = np.array(data2)
    x1_mean = x1.mean()
    x2_mean = x2.mean()
    d = np.sum(x1 - x2)**2
    D = np.sum((x1 - x2)**2)
    n = len(x1)
    df = len(x1) - 1
    var_d = np.sqrt((n*D-d)/df)/ np.sqrt(n)
    tstat = (x1_mean - x2_mean - mu)/(var_d/np.sqrt(n))
    if alternative in ["two-sided", "2-sided", "2s"]:
        pvalue = stats.t.sf(np.abs(tstat), df) * 2
    elif alternative in ["larger", "l"]:
        pvalue = stats.t.sf(tstat, df)
    elif alternative in ["smaller", "s"]:
        pvalue = stats.t.cdf(tstat, df)
    else:
        raise ValueError("invalid alternative")
    return tstat, pvalue


# In[26]:


pair_ttest(x1, x2, mu=8, alternative='smaller')


# # 4. Chi-squared test
# The usage of [chi-square test](https://en.wikipedia.org/wiki/Chi-squared_test):
# - Comparing the propotions of two or more populations
# - Independence testing between qualitative variables
# 
# Assumptions:
# - Populations are normally distributed

# In[10]:


import math
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings("ignore")


# In[5]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# ## 4.1. Dependent chi-squared test
# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# Is there a relationship between <code style='font-size:13px'>age_group</code> and <code style='font-size:13px'>degree</code>?
# 
# The hypotheses:
# - $H_0:$ The two variables are independent
# - $H_1:$ The two variables are dependent
# 
# <code style='font-size:13px'>age_group</code> and <code style='font-size:13px'>degree</code> are said to be strongly related if p-value $<0.05$.

# In[6]:


table = pd.crosstab(df.age_group,df.degree)


# In[7]:


chi, pvalue, dof, _ = stats.chi2_contingency(table)
print("chi stats:", chi)
print('p-value:', pvalue)


# In[11]:


expected, observed, summary = pg.chi2_independence(data=df, x='age_group', y='degree')


# In[9]:


expected


# In[33]:


summary


# ## 4.2. Proportion chi-squared test
# In R, Yate's correction chi-squared test is used for the <code style='font-size:13px'>prop.test</code> function. A Pearson's chi-squared is upward bias for 2x2 contingency table - an upwards bias tends to make results larger than they should be so Yate's correction is a regularization term in the formula of chi-squared statistic. However, Yate correction shouldn't be used because the correction is too strict for making the decision on data.
# 
# In that case, <code style='font-size:13px'>stats.chisquare</code> function can be used for proportion chi-square

# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# The number of officers is equal to the number of salespersons and is 5 times greater than the number of managers, true or false?
# 
# - $H_0: p_1=1/11, p_2=p_3=5/11$
# - $H_1$: There is at least one incorrect equation.

# In[34]:


df_chi = df.groupby('job').count()[['id']].reset_index()

df_chi['obs'] = df_chi.id/len(df)
df_chi['exp'] = [1/11,5/11,5/11]


# In[35]:


df_chi


# In[36]:


stats.chisquare(df_chi.obs,df_chi.exp)


# # 5. ANOVA
# [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) (Analysis of Variance) is a technique involving a collection of statistical tests analyzing the difference of the means of two or more groups. The means is calculated from a quantitative variable; the groups are determined using qualitative variables.

# ## 5.1. One-way ANOVA
# Usage: Compare multiple population means when there is a categorical variable containing at least three categories.
# 
# Assumptions:
# - Populations are normally distributed
# - Samples are random
# - Homogeneity of variances
# 
# The work flow:
# 1. Test the homogeneity of variances, using one of the following tests:
#     - [Bartlett's test](https://en.wikipedia.org/wiki/Bartlett%27s_test) (<code style='font-size:13px'>scipy.stats.bartlett</code> function)
#     - [Levene's test](https://en.wikipedia.org/wiki/Levene%27s_test) (<code style='font-size:13px'>scipy.stats.levene</code> function)
#     - Fligner-Killeen test (<code style='font-size:13px'>scipy.stats.fligner</code> function)
# 2. Test the equality of population means:
#     - If the variances are equal, use the <code style='font-size:13px'>scipy.stats.f_oneway</code> or <code style='font-size:13px'>pg.anova</code> function
#     - If the variances are not equal, use the <code style='font-size:13px'>pg.welch_anova</code> function
# 3. Post-hoc test to compare pairwise population means:
#     - If the variances are equal, use [Tukey's HSD test](https://en.wikipedia.org/wiki/Tukey%27s_range_test) (<code style='font-size:13px'>pg.pairwise_tukey</code>)
#     - If the variances are not equal, use Games-Howell test (<code style='font-size:13px'>pg.pairwise_gameshowell</code>)

# In[37]:


import math
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg


# In[38]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# <b style='color:navy'><i class="fa fa-book"></i>&nbsp; Case study</b><br>
# Is there a difference in average income between 3 areas (Central, Southern and Northern)? If there is, which group differs from the others?

# *Step 1*: Check the equality of population variances. If p-value $<0.05$, then reject $H_0$. The hypotheses:
# - $H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$
# - $H_1$: Exist at least one pair $\sigma_i^2 \neq \sigma_j^2 $ where $i \neq j$

# In[39]:


central = df[df['area'] =='central']['income']
northern = df[df['area'] =='northern']['income']
southern = df[df['area'] =='southern']['income']


# In[40]:


stats.bartlett(central, northern, southern)


# *Step 2*: Test whether the population means are equal or not. If p-value $<0.05$, then reject $H_0$. The hypotheses:
# - $H_0$: $\mu_1 = \mu_2 = \dots = \mu_k$
# - $H_1$: There is at least one pair $\mu_i \neq \mu_j $ where $i \neq j$

# In[41]:


stats.f_oneway(central, northern, southern)


# In[42]:


pg.anova(data=df, dv='income', between='area')


# In[43]:


pg.welch_anova(data=df, dv='income', between='area')


# *Step 3*: Post-hoc test to compare pairwise means. Any pair having p-value $<0.05$ can be considered significantly different in mean. 

# In[44]:


pg.pairwise_tukey(data=df, dv='income', between='area')


# In[45]:


pg.pairwise_gameshowell(data=df, dv='income', between='area')


# # 6. Distribution test
# The [Kolmogorov-Smirnov test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (KS test) is used to test whether a random variable follows a specific distribution or not. The test statistic is calculated as the difference between the empirical CDF of the observed variable and the CDF of the reference distribution. In SciPy, the `stats.ks` function performs Kolmogorov-Smirnov test.
# 
# Here are the popular distributions that `stats.ks` supports:
# 
# Distribution|function|Parameters         |
# :-----------|:---------|:------------------|
# Binomial    |`binom`  |`size`, `prob`     |
# Poisson     |`poisson`   |`lambda`           |
# Unifrom     |`uniform`   |`min`, `max`       |
# Normal      |`norm`   |`mean`, `sd`       |
# Cauchy      |`cauchy` |`location`, `scale`|
# T           |`t`      |`df`               |
# F           |`f`      |`df1`, `df2`       |
# Chi-squared |`chi`  |`df`               |
# Beta        |`beta`   |`shape1`, `shape2` |
# Gamma       |`gamma`  |`shape`, `scale`   |

# In[46]:


import math
import numpy as np
import pandas as pd
from scipy import stats


# In[47]:


df = pd.read_csv('data/hypothesis.csv')
df.head()


# In[48]:


stats.kstest(df.age, cdf = 'norm', args=(24, 0.05))


# ---
# *&#9829; By Quang Hung x Thuy Linh &#9829;*
