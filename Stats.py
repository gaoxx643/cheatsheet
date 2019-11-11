## resampling
# resample with replacement
resample_risk = df.query('risk==1').sample(5000, replace=True)

# resample withour replacement
resample_safe = df.query('risk==0').sample(5000, replace=False)

# randomly sampling with replacement
df.sample(n=100, replace=True, random_state=1)

df.sample(n=100, replace=False, random_state=1)

# stratified sampling
df.sample(n=100, weights='size',random_state=1)

# bootstrapping
def bootstrap_resample(x, n=None):
	if n = None:
		n = len(x)

	resample_i = np.floor(np.random.rand(n)*len(x)).astype(int)
	x_resample = x[resample_i]
	return x_resample

===============================================================================================================
## skewness and normaliztion (--数据常用方法： log, scale等(需要： 基于参数的模型或基于距离的模型，都是要进行特征的归一化; 不需要：基于树的方法是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等)
# check normorality
stat, p = stats.normaltest(df.driver_robot_entrance_count)
print('Statistics=%.3f, p=%.3f' % (stat,p))
alpha=0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussin (reject H0)')


# check Kurtosis of normal distribution
stats.kurtosis(df.driver_count)

# check skewness of normal distribution
stats.skew(df.driver_count)


# normalize with scale [0,1]
from sklearn import preprocessing as prep
count_scaled = prep.scale(df['count'])

# normalize with log [-1,1]
train['price'] = np.log1p(train['price'])

# normalize with minmax [0,1]
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)


# standardize data with standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.fit_transform(x_test)


# log transform skewed numeric features
numeric_feats = all_data.dtypes[all_data.dtypes!='object'].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats>0.75]
skewed_feats_index = skewed_feats.index

all_data[skewed_feats_index] = np.log1p(all_data[skewed_feats_index])
all_data = pd.get_dummies(all_data)

===============================================================================================================
## stats function
def perm_func(df,n1,n2):
	N = n1 + n2
	idx_b = random.sample(range(n), n1)
	idx_a = list(set(range(n)) - set(idx_b))
	mean_diff = df.ix[idx_b].mean() - df.ix[idx_a].mean()
	return mean_diff

temp = {'id':[], 'value':[]}
for i in range(1000):
	cal = perm_func(df,139,53)
	temp['id'].append(i)
	temp['value'].append(cal)

df = pd.DataFrame.from_dict(temp)


# hyper-geometric distribution
def get_odds(m,n):
	M = m # samples drawn from the populations
	N = n # size of populations
	df = []
	for i in [0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
		n = math.cell(M * i)
		rv = hypergeom(M,n,N)
		x = np.arange(0,n+1)
		pmf_black = rv.pmf(x)
		pdf = pd.DataFrame(pmf_black)
		pdf.columns = ['odds']
		pdf['acc_odds'] = (1-pdf['odds'].cumsum())
		df.append([i, pdf['accu_odds'][29]])
		odds_table = pd.DataFrame(df,columns=['% of black balls', odds to draw at least 29 black balls])
	return odds_table

# conduct t-test on two independent samples
def independent_ttest(data1,data2,alpha):
    from scipy.stats import sem
    from scipy.stats import t
    from numpy import mean
    from math import sqrt
    
    # calcualte means
    mean1,mean2 = mean(data1),mean(data2)
    # caculate standard errors
    se1, se2 = sem(data1),sem(data2)
    # standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1)+len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return results
    return t_stat, df, cv, p

 # conduct proportion z-test on two independent samples
def get_proportion_ztest(x1,y1,x2,y2):  # x1/y1 is prob of event 1 while x2/y2 is prob of event 2
    count = np.array([x1,x2])
    nobs = np.array([y1,y2])
    stat, pval = proportions_ztest(count,nobs)
    print('z-stat is {0:0.3f} &'.format(stat),'p-value is {0:0.3f}'.format(pval))


