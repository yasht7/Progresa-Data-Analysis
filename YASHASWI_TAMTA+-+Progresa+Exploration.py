
# coding: utf-8

# # Exploration of the Progresa Data set
# 
# This will involve the explorations and analysis of the progresa data set in an attempt to quantize the evident effect of the program if any.

# ## Introduction
# 
# We will be using data from the [Progresa program](http://en.wikipedia.org/wiki/Oportunidades), a government social assistance program in Mexico. This program, as well as the details of its impact, are described in the paper "[School subsidies for the poor: evaluating the Mexican Progresa poverty program](http://www.sciencedirect.com/science/article/pii/S0304387803001858)", by Paul Shultz (available on Canvas). This will be used to familiarize ourself with the progresa dataset and get a rough sense of where the data comes from and how was it generated.
# 
# The goal of this exploration is to implement some of the basic econometric techniques learnt in Machine Learning class to measure the impact of Progresa on secondary school enrollment rates. The timeline of the program was:
# 
#  * Baseline survey conducted in 1997
#  * Intervention begins in 1998, "Wave 1" of data collected in 1998
#  * "Wave 2 of data" collected in 1999
#  * Evaluation ends in 2000, at which point the control villages were treated. 
#  
# In the data, each row corresponds to an observation taken for a given child for a given year. There are two years of data (1997 and 1998), and just under 40,000 children who are surveyed in each year. For each child-year observation, the following variables are collected:
# 
# | Variable name | Description|
# |---------|---------|
# |year	  |year in which data is collected
# |sex	  |male = 1|
# |indig	  |indigenous = 1|
# |dist_sec |nearest distance to a secondary school|
# |sc	      |enrolled in school in year of survey|
# |grc      |grade enrolled|
# |fam_n    |family size|
# |min_dist |	min distance to an urban center|
# |dist_cap |	min distance to the capital|
# |poor     |	poor = 1|
# |progresa |treatment =1|
# |hohedu	  |years of schooling of head of household|
# |hohwag	  |monthly wages of head of household|
# |welfare_index|	welfare index used to classify poor|
# |hohsex	  |gender of head of household (male=1)|
# |hohage   |age of head of household|
# |age      |years old|
# |folnum	  |individual id|
# |village  |	village id|
# |sc97	  |schooling in 1997|

# ---
# 
# ## Part 1: Descriptive analysis
# 
# ### 1.1	Summary Statistics
# 
# Presenting summary statistics (mean and standard deviation) for all of the demographic variables in the dataset (i.e., everything except year, folnum, village). Present these in a single table alphabetized by variable name.

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
get_ipython().magic('matplotlib inline')

progresa_df = pd.read_csv('progresa_sample.csv')

# Cleaning the data so it fits the description
progresa_df['poor'] = progresa_df['poor'].map({'pobre': 1, 'no pobre': 0})
progresa_df['progresa'] = progresa_df['progresa'].map({'basal' : 1, '0':0})

# Generating a Dataframe consisting the mean and standard deviation for each required variable.
# sorting the columns and storing them in a new dataframe
summary = progresa_df.reindex_axis(sorted(progresa_df.columns), axis=1)


# Method1: Inversing the Dataframe and then creating a new column.
summary = summary.transpose()
# Removing non-demographic columns
summary.drop(summary.index[[list(summary.index).index("year"), list(summary.index).index("folnum"), list(summary.index).index("village")]], inplace=True)

# Calculating Mena and Standar Deviation
summary['Mean'] = summary.mean(axis=1)
summary['Std'] = summary.std(axis=1)

# deleting multiple columns
summary.drop(summary.columns[:77250], inplace=True, axis=1)
summary


# In[2]:

len(progresa_df['village'].unique())


# Initial observation from the Data:
# These columns are to be refactored to match the decribed values:
#     * 'poor' : pobre(spanish for poor) -> 1 and no pobre -> 0. This indicates whether the 
#     * 'progresa' : basal -> 1 and 0 is as it is. This indicated whether the data is in treatement or control.
#     
# From here there are two way to proceed
#     * Take the transpose of the dataframe and calulate the mean of required columns.
#     * Slicing and storing the columns values as rows.
# We have gone for the first method as it is memory conservative and the original Dataframe is unadulterated. Removing the non-demographic columns using their indices. Caluculating Mean and standard deviation as new columns and then removing the remaining 77250 data columns.

# ### 1.2 Differences at baseline
# 
# Are the baseline (1997) demographic characteristics **for the poor**  different in treatment and control villages? Using a T-Test to determine whether there is a statistically significant difference in the average values of each of the variables in the dataset. Here we focus only on the data from 1997 for individuals who are poor (i.e., poor=='pobre').
# 
# Results presented in a single table with the following columns and 14 (or so) rows:
# 
# | Variable name | Average value (Treatment villages) | Average value (Control villages) | Difference (Treat - Control) | p-value |
# |------|------|------|------|------|
# |Male  |?     |?     |?     |?     |
# 

# In[3]:

from scipy import stats
from scipy.stats import t


# In[4]:

# Segregating into treatment and Control data
treatment_97 = pd.DataFrame(progresa_df[(progresa_df.year == 97) & (progresa_df.poor == 1) & (progresa_df.progresa == 1)])
control_97 = pd.DataFrame(progresa_df[(progresa_df.year == 97) & (progresa_df.poor == 1) & (progresa_df.progresa == 0)])

treatment_97.shape


# In[5]:

# Selecting rows where poor=1 and the year=97, and then grouping by 'progresa' column
ttest_df = progresa_df[(progresa_df.year == 97) & (progresa_df.poor == 1)].groupby('progresa').mean()
ttest_df.drop(ttest_df.columns[[0,9,16,17]], axis =1,inplace=True)

ttest_df = ttest_df.transpose()
# swapping columns to match the structure of the required table
ttest_df = ttest_df[[1,0]]

# Resetting Index
ttest_df.reset_index(level=0, inplace=True)
ttest_df.rename(columns={'index' : 'Variable name', 0: 'Average value (Control villages)', 1: 'Average value (Treatment villages)'}, inplace=True)

# This can be used to null the column name for a much cleaner output.
# ttest_df.columns.names = ['']

# List of all Variables
var_list = list(ttest_df['Variable name'])

# Calculating T test for the Treatment, Control
tt = list(stats.ttest_ind(treatment_97[var_list], control_97[var_list], nan_policy='omit'))

# Adding the remaining two columns.
ttest_df['Difference (Treat - Control)'] = tt[0]
ttest_df['p-value'] = tt[1]

# for a better look at the insignificant data with respect to the value of p
# ttest_df['p<0.05'] = ttest_df['p-value'] < 0.05

ttest_df


# Here we're applying a double sided t test, in an attempt to unearth any significant statistical diffrences between control and treatment groups.
# 
# Analysis:
# 
# **Aim:**
#     To check for significant statistical differences between the baseline of both treatment and control groups before the treatment began.
# 
# **How were the groups selected:**
#     *source* - Translated from Spanish. SEDESOL, Más Oportunidades para las Familias pobres - Primeros Avances, 393-394z
#     
#     "The design of the impact evaluation of Progresa in communities and households is quasi-experimental...To undertake this component of the evaluation, a random sample of communities with 'high' or 'very high' degrees of marginalization were selected which would be incorporated into the program during Phase II (November 1997) and which would serve as the [treatment] communities...Another sample of communities with similar characteristics was randomly designated from those that could have been the object of later selection, and that could function as controls...the size of the sample was estimated starting from a universe of 4,546 localities to choose 330 base localities, and from a universe of 1,850 to choose 191 control localities, using a distribution proportional to the size of the locality."
# 
#  From here we can observe that the treatment and control groups were randomly segregated.
# 
# **T Test:**
#     The output of this test are the t-value and the p-value.
#     *T value* can be labeled as a simple signal to noise ratio. When the t value is greater than the critical value of the data we then, reject the null hypothesis which deems no statistical difference between the two samples/vriables. Vice-versa when its less that the critical value.
#     
# **Conclusion:** the following variables have displayed statistically significant differences in average values at the baseline: *Sex* (gender), *dist_sec* (nearest distance to a secondary school), *min_dist* (min distance to an urban center), *dist_cap* (min distance to the capital), *hohedu* (years of schooling of head of household), *hohwag* (monthly wages of head of household), *welfare_index* (welfare index used to classify poor), *hohage* (age of head of household). As these have p values < 0.05.
# 

# ### 1.3 Interpretation

# **A: Are there statistically significant differences between treatment and control villages as baseline?** 
# Observing the output of the T-Test the statistical dofferences are eveident between the two samples of treatment and control. Especially in the following variables.
#     - Sex, dsit_sec, min_dist, dist_cap, hohedu, hohwag, welfare_index, hohage
# 
# **B: Why does it matter if there are differences at baseline?**
# The differences in Baseline tell us that the samples, which were randomly picked and segregated between Treatment and Control were not perfectly random and were statistically different from each other. The basis of Treatment and control strategy is to observe the effect of a particular treatment on a group of indivisuals vs those which did not recieve the treatment. This requires us to start at an indistinguishable baseline in order to record accurate causal analysis and observe genuine patterns.
# 
# **C: What does this imply about how to measure the impact of the treatment?**
# From the above analysis we can observe that the two groups are statistically different even before the treatment commences. Therefore, we can assert that the segreagation performed randomly is flawed. Therefore we should come up with some other alternative which will help us segregate the population into two statistically indifferent groups. Therfore, instead of randomizing the segregation, we can use *Stratified Sampling*. This method can definitely help us achieve the required precision with the same sample size.

# ### 1.4 Graphical exploration, part 1
# 
# For each level of household head education, computing the average enrollment rate in 1997. Creating a scatterplot that shows this relationship.

# In[6]:

# calculating the enrollment rate.
enrate = pd.DataFrame(progresa_df[(progresa_df.year == 97)].groupby('hohedu').mean()['sc'])

# calculating the frequency of observations for each level of household education.
countrate = pd.DataFrame(progresa_df[(progresa_df.year == 97)].groupby('hohedu').count()['sc'])
countrate.rename(columns={'sc':'count'}, inplace='True')
countrate.reset_index(level=0, inplace=True)
enrate.reset_index(level=0, inplace=True)

# merging the data frames
enrate = enrate.merge(countrate, on='hohedu')

print(enrate)

# plotting
matplotlib.style.use('ggplot')

plt.scatter(list(enrate['hohedu']), list(enrate['sc']), c=list(enrate['count']), s=50, edgecolor='blue')
plt.title('Average enrollment rate Vs Household education')
plt.xlabel('Number of household education years')
plt.colorbar()
plt.ylabel('Average Enrollment rate')


# In[7]:

progresa_df[(progresa_df.hohedu==20)].sc


# The plot obtained above alomst conforms to the general norm of an increasing relationship. I have also considered the frequency of the data as well and mapped the scatter density accordingly. Higher frequency, higher the density, and darker the color. This helps us to get a general sense of data and disregard outliers when making conlcusions. The increasing nature of the plot drops at points 10, 14 and then drastically at 20. Referring to the count curve we can see that the point 20 has only 2 data points in the average compared to 9000+ of 0 years. On further expounding on the data we found that we have 4 values corresponding to the 'hohedu' at 20. Out of which 2 are NaN and one is 0 and the other is 1. SO the data points are not sufficient enough to be considered.
# As the years increase we can also see the decrease in the number of values. Therfore, we can set up a threshold to be near the mode of the data, instead of the mean and disregard any count value less than this threshold.
# Doing this, we can conclude, mostly an increasing trend throughout, and strictly till the household education years equals 10. But the maximum value is obtained at 18 years

# ### 1.5 Graphical exploration, part 2
# 
# Creating a histogram of village enrollment rates **among poor households in treated villages**, before and after treatment. Specifically, for each village, calculating the average rate of enrollment of poor households in treated villages in 1997, then compute the average rate of enrollment of poor households in treated villages in 1998. Create two separate histograms showing the distribution of these average enrollments rates, one histogram for 1997 and one histogram for 1998. On each histogram, draw a vertical line that intersects the x-axis at the average value (across all households).
# 

# In[8]:

# Preparing the Data for plot.
# Before treatment data
before_t = pd.DataFrame(progresa_df[(progresa_df.year == 97) & (progresa_df.progresa == 1) & (progresa_df.poor ==1)].groupby('village').mean()['sc'])
before_t.reset_index(level=0, inplace=True)
# After treatment data
after_t = pd.DataFrame(progresa_df[(progresa_df.year == 98) & (progresa_df.progresa == 1) & (progresa_df.poor ==1)].groupby('village').mean()['sc'])
after_t.reset_index(level=0, inplace=True)


fig = plt.figure()

# Histogram before the Treatment
ax1 = fig.add_subplot(121)
ax1.hist(before_t['sc'], color = "purple", edgecolor='gold')
xlabel("Avg Enrollment Rates in 1997")
ylabel("Number of Villages")
# setting the limits for axes, for better comparision
ax1.set_ylim([0,90])
ax1.set_xlim([0.4,1.1])
plt.axvline(before_t['sc'].mean(), color='gold', linestyle='dashed', linewidth=3)


# Histogram before the Treatment
ax2 = fig.add_subplot(122)
ax2.hist(after_t['sc'], color = 'gold', edgecolor='purple')
xlabel("Avg Enrollment Rates in 1998")
# setting the limits for axes, for better comparision
ax2.set_xlim([0.4,1.1])
plt.axvline(after_t['sc'].mean(), color='purple', linestyle='dashed', linewidth=2)
plt.show()

print()
print("The mean enrollment BEFORE treatment is: ", before_t['sc'].mean())
print("The mean enrollment AFTER treatment is: ", after_t['sc'].mean(), "\n")

# just like above we can see the differences in the data. using a T test.
print("The output of T test on the above values are:\n", stats.ttest_ind(before_t['sc'], after_t['sc'], nan_policy='omit'))


# Here is the result, which will tell us what effect did the treatment have on the sample population. 
# First set of observations include 
#     - The noticable dip in number of villages with 'average enrollment rate' between 0.65 and 0.75. 
#     - This is accompanies by an increase in villages with 0.4 'average enrollment rate' from 0 to 3.
#     - A noticable increase in #villages with 'avg enrol rate' > 0.75
# Now, the *p-value* obtained from the t test is 0.04 < 0.05. This means that there is a statitical difference between them. But given that, difference is the measure of efectiveness of the treatment, we should be expecting a much better *p-value* which is farther from the 0.05 mark than the one obtained.

# ## Part 2: Measuring Impact
# 
# ### 2.1 Simple differences: T-test
# 
# Estimating the impact of Progresa using "simple differences." Restricting to data from 1998 (after treatment).
# 
# * calculating the average enrollment rate among **poor** households in the Treatment villages and the average enrollment rate among **poor** households in the control villages.
# * Determining which difference estimator in Schultz (2004) does this approach correspond to?
# * Using a t-test to determine if this difference is statistically significant.

# In[9]:

en_rate = pd.DataFrame(progresa_df[(progresa_df.poor==1) & (progresa_df.year==98)].groupby('progresa').mean()['sc'])
print('The average enrollement rate for poor households in the year 98 is as follows:')
print('Treatment villages: ', en_rate.sc[1])
print('Control villages: ', en_rate.sc[0])
en_rate.reset_index(level=0, inplace=True)

# Applying T-test to measure the statistical difference
treatment_98 = pd.DataFrame(progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==1) & (progresa_df.year==98)])
control_98 = pd.DataFrame(progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==0)& (progresa_df.year==98)])

tt2 = stats.ttest_ind(treatment_98['sc'], control_98['sc'], nan_policy='omit')
print('\nThe output of T test is as follows:\n', tt2)


# The difference estimate used in Schultz(2004) is a Difference in difference estimator also called diff-in-diff. In this observation we take in the differences in two samples, treatment and control, between intervals before and after. This way we can eradicate any differences that occured in the samples regardless of any intervention.
# 
# The obtained p value is very much less than 0.05 therfore, the difference is not much significant.

# ### 2.2 Simple differences: Regression
# 
# Estimating the effects of Progresa on enrollment using a regression model, by regressing the 1998 enrollment rates **of the poor** on treatment assignment.

# In[10]:

import statsmodels.formula.api as smf
en_regr = progresa_df[(progresa_df.year==98) & (progresa_df.poor ==1)]

regr = smf.ols('sc ~ progresa', data= en_regr).fit()
regr.summary()


# **Observations:**
# 
# **Based on this model, how much did Progresa increase the likelihood of a child enrolling?**
#     We can observe that the parameter associated with progresa is positive, hence an increasing relationship. The magnitude of this parameter found is 0.0388, the magnitude of the intercept is found to be 0.8076. Therefore Progresa increases the likelihood of enrollment by 0.0388 i.e. likelihood of enrollment without progresa is 0.8076 and with progresa is 0.8464
# 
# **How does your regression estimate compare to your t-test estimate from part 2.1?**
#     The regression model here validates the values obtained in 2.1. Avg. enrolleent rate of control villages(0.80) is the same while the avg. enrollment rae of treatment, according to the model is given as 0.80 + 0.0388 = 0.84, which is same as well.
# 
# **Based on this regression model, can we reject the null hypothesis that the treatment effects are zero?**
#     No. The regression model tells us that the p value obtained is not greater than 0.05 hence we cannot reject the null hypothesis.
# 
# **What is the counterfactual assumption underlying this regression?**
#     The counter factual assumption underlying this regression is that the avg. enrollement rate of the treatment villages would be same as that in the control villages.

# In[11]:

## LINEAR REGRESSION IN DEPTH.!!!
# a small test to see the imapct of a completrly opposite variable on the values of 'sc'
test_df = progresa_df.copy()
test_df['v1'] = test_df.sc == 0
test_df['v3'] = test_df.sc == 1

# v1 has the opposite value of sc.
test_df['v1'] = test_df['v1'].map({True : 1, False:0})
# v3 has the same value of sc.
test_df['v3'] = test_df['v3'].map({True : 1, False:0})

# Regression test
test_regr = smf.ols(formula='sc ~ v1 + v3', data=test_df).fit()
test_regr.summary()


# ### Self intuition.
# 
# We see here that the coefficient of v1 is -0.3333 and v3 is 0.6667 and the intercept is 0.3333. confirming that negative relation actually means an opposite nature.!

# ### 2.3 Multiple Regression
# 
# Re-comuting the above regression estimated by including a set of control variables. Include, for instance, age, distance to a secondary school, gender, education of household head, indigenous, etc.
# 

# In[12]:

multi_regr = smf.ols('sc ~ progresa + age + dist_sec + sex + hohedu + indig', data=en_regr).fit()
multi_regr.summary()


# **Observations:**
# 
# **How do the controls affect the point estimate of treatment effect?**
#     The very first effect we can see is the decrease in point estimate of the treatment from 0.00388 ot 0.00356. We can also observe that there are other parameters which exhibit statistically significant effect on the avg. enrollment rate.
# 
# **How do the controls affect the standard error on the treatment effect?**
#     The standard error on the treatment has dropped from 0.05 to 0.04, which resonates with the validity of null hypothesis as, with a samller error we have a smaller point estimate of the treatment effect.
# 
# **How do you interpret the differences (or similarities) between your estimates of 2.2 and 2.3?**
#     By inclusion of more control variables, the multivariate regression model conform to the prediction of the previous univariate regression model with even better std. error rate. Thus, the inclusion of control varialbles have definitely increased the confidence of accepting the null hypothesis.
#     Assessing the affect of control variables, Apart from receiving the funds from progresa, we can see that natives are most likely to enroll their kids into a school. This is understood as the natives tend to know their neighbourhood better adn would definitely socialize more. Moreover, we cannot ignore the negative relation of enrollment rate with the child's age. It decreases by 0.0655 for every increment in age. This would mean that the older childeren are less likely to enroll in a school. The only other variable which demonstrates an inverse relation is, without a surprise, distance to secondary school. Lastly, it;s very interesting to see the high value of intercept, almost twice when compared to 2.2 and with similar in it's std. error.

# ### 2.4 Difference-in-Difference, version 1 (tabular)
# 
# Thus far, we have computed the effects of Progresa by estimating the difference in 1998 enrollment rates across villages. An alternative approach would be to compute the treatment effect using a difference-in-differences framework.
# 
# Estimating the average treatment effects of the program for poor households using data from 1997 and 1998. Specifically, calculate the difference (between 1997 and 1998) in enrollment rates among poor households in treated villages; then computing the difference (between 1997 and 1998) in enrollment rates among poor households in treated villages. The difference between these two differences is our estimate.
# 

# In[13]:

# Splitting the control group.
en_control_97 = progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==0) & (progresa_df.year==97)] 
en_control_98 = progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==0) & (progresa_df.year==98)]

# SPlitting the treatment group.
en_treat_97 = progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==1) & (progresa_df.year==97)]
en_treat_98 = progresa_df[(progresa_df.poor==1) & (progresa_df.progresa==1) & (progresa_df.year==98)]

# Creating a new Dataframe
index = ['Control sample', 'Treatment sample']
cols = ['Avg. Enrollment rate in 97', 'Avg. Enrollment rate in 98']
new_diff = pd.DataFrame(index = index, columns = cols)

# filling in the values
new_diff.loc['Control sample', 'Avg. Enrollment rate in 97'] = en_control_97['sc'].mean()
new_diff.loc['Control sample', 'Avg. Enrollment rate in 98'] = en_control_98['sc'].mean()
new_diff.loc['Treatment sample', 'Avg. Enrollment rate in 97'] = en_treat_97['sc'].mean()
new_diff.loc['Treatment sample', 'Avg. Enrollment rate in 98'] = en_treat_98['sc'].mean()

new_diff['Difference'] = new_diff['Avg. Enrollment rate in 98'] - new_diff['Avg. Enrollment rate in 97']

print("The Diff-in-Diff estimate is: ", new_diff.Difference[1] - new_diff.Difference[0])

print("\nThe Tabular representation of the Differences:")
new_diff


# **Observations:**
# 
# **What is your estimate of the impact, and how does it compare to your earlier (simple difference) results?**
#     The estimate of the impact is 0.0313. This indicates that the increase in the enrollment rate can be credited to the progresa treatment. The diff-in-diff estimate is lower than the earlier observed point estimate of 0.0388(univariate) and 0.0356(multivariate). Since, the lower multivariate estimate had a better std. error, we could say that the diff-in-diff estimate is more accurate
# 
# **What is the counterfactual assumption underlying this estimate?**
#     The counterfactual assumption would be that the avg. enrollment change in the treatment group and control group would be same irrespective of the treatment.

# ### 2.5 Difference-in-Difference, version 1 (regression)
# 
# Using a regression specification to estimate the average treatment effect of the program (on the poor) in a difference-in-differences framework. Selecting 5 control variables.
# 
# 

# In[14]:

# TO get a separate statistic with respect to the year we create a dependent/correlated variable: afterT (1= after treatment).
lm_diff = progresa_df[(progresa_df.poor==1)]
lm_diff.loc[lm_diff.year==97, 'afterT'] = 0
lm_diff.loc[lm_diff.year==98, 'afterT'] = 1

# running a regression model, we'll chose the previously chosen control variables to make the intercept comparable
ddregr = smf.ols(formula='sc ~ progresa + afterT + progresa:afterT + age + dist_sec + sex + hohedu + indig', data =lm_diff).fit()
ddregr.summary()


# **OBSERVATIONS:**
# 
# In order to run the regression test in the diff-in-diff framework we have to consider values of  the feature 'year' interactively with the feature 'progresa' i.e in year=97 vs year=98. We can simply assign 1 and 0 for the year 98 and 97 respectively. This would give us the before and after period. Earlier we considered only the data in the year 1998 which is post treatment.
# 
# **What is your estimate of the impact of Progresa? Be very specific in interpreting your coefficients and standard errors, and make sure to specify exactly what units you are measuring and estimating.**    
#     The first and foremost area of interest is the interation variable of year and treatment, progresa:afterT. The value of this estimate is 0.0314 with a standard error of 0.06. This increases the avg enrollement rate by 0.0314. We can also see that the effect of progresa variable is diminished to 0.004. Thereforem the estimate of th impact of progresa will be **0.0043 + 0.0314 x afterT**.
#     Earlier we compared the effect of progresa in the year 98 which was after treatment, but here, we obtained an estimate by considering the data from both the years and including both the groups of treatment and control. We measured the trends in avg.  enrollment rate with respect to the presence of treatment, and generalized the change as a 3.14% increase. We can also observer here that the combined effect of progresa adn the year has amuch greater impact than progresa in the same model.
# 
# **How do these estimates of the treatment effect compare to the estimates based on the simple difference?**
#     The estimate of simple difference was found to be 0.0388 with a standard error of 0.05 whereas the estimated diference found out here using the diff in diff mehtod is lower. We could say the the simple differences has given us a higher value with lesser accuracy and the current method outputs a more accurate lower value.(The current value's accuracy is credited to the better std.  error)
# 
# **How do these estimates compare to the difference-in-difference estimates from 2.4 above? What accounts for these differences?**
#     The estimate of the the effect of progresa on the population, by the regression model, is given by the sum of all the values of the coefficient influencing the contol variable 'progresa'. In the previous model we calculated the differences in the both treatment and control group before and after the treatment commenced. The differences in the values isn't significant but in the current method we've considered other control vairables as well as an interation variable of 'progresa' and 'year', which puts us in a more accurate and inclusive estimate.
# 
# **What is the counterfactual assumption underlying this regression?**
#     The counter factual assumption would be that, The trends of avg enrollment rate on both the groups will be the same even without the treatment.

# ### 2.6 Difference-in-Difference, version 2
# 
# In the previous problem, we estimated a difference-in-differences model that compared changes in enrollment rates over time across treatment and control villages. An alternative approach would be to compare enrollment rates in 1998 between poor and non-poor across treatment and control villages. 
# 

# In[15]:

# Including the interation variable between the poor and progresa as well.
dd_lm2 = smf.ols(formula='sc ~ progresa + poor + progresa:poor + age + dist_sec + sex + hohedu + indig', data = progresa_df).fit()
dd_lm2.summary()


# **OBSERVATIONS:**
# 
# **How would we estimate this version of the treatment effects in a regression model?**
#     The effect of treatment, in this model, is given by the sum of coefficients influencing the 'profresa' variable. Here it is **0.027 - 0.008 x poor**, where poor is the value indicating whether the subject is labeed as poor or not.
#     
# **What is the counterfactual assumption underlying this regression?**
#     The counterfactual assumption underlying the regression is that, the changes in avg. enrollment rate is same for 'pobre'(poor) and 'no pobre'(not poor) households even in the absence of treatment.
#     
# **How do these treatment effects compare to the estimates above?**
#     The treatment effects of this estimate is much lower than the one in **2.5**. The combined estimate of the progresa on the avg. enrollment rate here is given by: **0.027 - 0.0080 x poor**. The coefficient of the interation variable of the treatment and the status of the households is negative and almost zero that woudl indicate almost an opposite impact of the variables on the avg. enrollment rate (Tested in the cell before 2.3). Whereas, in the previous estimate of **2.5** we have a much more realistic and accurate measure of the estiamte. 
#     
# **Discuss some possible explanations for differences or similarities**
#     This could be because there is no expected effect of the interation of these two variables. This conforms to the progresa model where only the poor households are recipient of the treatment. One other uncanny observation is the negative coefficent of 'poor' because, the household recieves the progresa treatment only when the poor value is 1. There is every reason for this coefficient to be positive, as we would expect that the increase in enrollment rate would be more in poor households.
# 

# ### 2.7 Spillover effects
# 
# Thus far, we have focused on the impact of PROGRESA on poor households. Repeating the analysis in 2.5, using a double-difference regression to estimate the impact of PROGRESA on non-poor households. In other words, comparing the difference in enrollments between 1997 and 1998 for non-poor households in treatment villages with the difference in enrollments between 1997 and 1998 for non-poor households in control villages.
# 

# In[16]:

# TO get a separate statistic with respect to the year we create a dependent/correlated variable: afterT (1= after treatment).
lm_diff2 = progresa_df[(progresa_df.poor==0)]
lm_diff2.loc[lm_diff2.year==97, 'afterT'] = 0
lm_diff2.loc[lm_diff2.year==98, 'afterT'] = 1

print("Total number of instances where non-poor subjects were also handed the treatment are: ", len(progresa_df[(progresa_df.poor==0) & (progresa_df.progresa==1)]))

# running a regression model, we'll chose the previously chosen control variables to make the intercept comparable
ddregr2 = smf.ols(formula='sc ~ progresa + afterT + progresa:afterT + age + dist_sec + sex + hohedu + indig', data =lm_diff).fit()
ddregr2.summary()


# **OBSERVATIONS:**
# 
# Here we get the analysis of the non poor households, we can see the diff in diff estimate of the effect progresa has on the avg, enrollment rate. The total estimate is **0.0043 + 0.0314 x afterT**. NOrmally we should not see any effect here, or see lesser effect but, the estimates here are closely relatable to the ones obtained before.
# 
# **A: Describe one or two reasons why PROGRESA might have impacted non-poor households.**
#     Firstly, one of the possibility can be the contamination of data i.e. the possibility of families migrating from the control locality to the treatment locality. This could increase their enrollment rate similar to that of treatment but the resulting data would show up on the control side. Second reason could be that the non-poor households could have been sending their childeren to school on noticing it's importance from the perspective of progresa program or could also be a socical competition. This could account towards the increase in the enrollment rate of non poor households.
# 
# **B: Do you observe any impacts of PROGRESA on the non-poor?**
#     We have the impact of progresa on the avg. enrollment rate quantified by the sum **0.0043 + 0.0314** when afterT is equal to 1. Therefore, we see a **0.0357** increase in the value of avg. enrollment rate solely due to the dependent coefficients of 'progresa'.
# 
# **C: What is the identifying assumption that you are using to estimate the impact of PROGRESA on non-poor households.**
#     The assumption here, that helps in identifying the estimate of impact of progresa on avg. enrollment rate is, that thfe progresa program could've in some way effected those household which were non-poor or, assuming that the effect, of progresa, would've anyways happened even without the treatment actually existing.

# ### 2.8 Summary
# 
# Based on all the analysis you have undertaken to date, do you thik that Progresa had a causal impact on the enrollment rates of poor households in Mexico?

# The progresa program seems to be very good and promising on the paper, moreover the intricacies involved in the project are unique and immune to any political intervention or corruption. Just looking at the final results we can see that there is a statistically significant increase in the enrollment rate across the entire population, treatment and control, poor and non-poor. But is the increase enough.? Now let's summarize the step by step analysis of the internal working of the program.
# 
# 1. The division of population into control and treatment group, even though random, had some statistically significant differences therfore the baseline assumption of both groups being similar was false. This makes the further analysis on those groups not perfectly reliable.
# 
# 2. Secondly, looking the avg. enrollment rate it has increased when we considered the impact by performing multivariate linear regression on progresa, we observed an even more accurate estimate.
# 
# 3. Then we used the Difference in difference mehtod, which even considers the counter facutal assumption of the control group. The estimate dropped. and one performing linear regression in the Diff-in-Diff model we observed that the combined impact of the treatment and the year was much more higher than either alone.
# 
# Given these above tests whcih were primal of all conducted, The increase in enrollment exists but, in order to assert fair judgement on the Progresa plan we would need some economic details which would give us insight into the profit and loss data. Considering the monetary aspect of any program is paramount, althogh in social terms it can be compared to the amount of effort put into the program and the social benefits recieved from it. This being said, from the analysis it appears that the turnaround of the enrollment rate doesn't do justice to all the amount of subsidy provided to each family. There could be some families that recieved the subsidy but did not enroll their kids into the schools.
# There's another possibility that the amount of subsidy wasn't enough in terms of quantity because, looking at the data, some of the companies were earning really low and the subsidy would've just helped them making ends meet this could've compelled their childeren to avoid school in order to reduce financial stress.
