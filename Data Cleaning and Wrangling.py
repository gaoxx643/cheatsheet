##  data glimpse

# overview of numerical fields
data.describe() 

# overview of object fields
data.describe(include=['O']) 

# check number of rows and cols
data.shape 

# check the first rows
data.head() 

# check the last rows
data.tail() 

# explore the df on min, max, std, missing
def explore_df(df):
	explore = df.describe(inlcude='all').T
	explore['null'] = len(df) - explore['count']
	explore['null_percent'] = explore['null']/len(df)
	explore.insert(0,'dtype',df.dtypes)
	explore.reset_index(inplace=True)
	explore.sort_values(by=['dtype','null_percent'],ascending=[False,False],inplace=True)
	explore.reset_index(drop=True, inplace=True)
	return explore

# examine the imbalanced degree of a df
def imbalanced_degree(df):
	res = []
	for col in df.columns:
		res.append({'feature': col,
					'imbalanced_degree': df[col].value_counts().max()/len(df[col])})
		temp = pd.DataFrame(res)
		res = temp.sort_values(by='imbalanced_degree',ascending=False)

	return temp
================================================================================================

## data formating
# display 3 digits only
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df.describe()

# keep two digits
format = lambda x: '%.2f' % x
df.applymap(format)

# keep two digits via round()
data['price'] = round(df['price'],2)


## indexing handling
# assign index with year
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1700','2008')) # need to import statesmodels.api as sm

# assign index by quarter
index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1','2009Q3')) # need to import statesmodels.api as sm
data.index = pd.DatetimeIndex(data.date, freq='QS')

# set index
df = pd.Series(np.arange(5),index=['a','b','c','d','e'])

# set index
data.set_index('column_name',inplace=True)

================================================================================================

## merge and join data tables
# inner join 
df1.merge(df2,how='inner',on='id')

df1.merge(df2,how='inner',left_on='id',right_on='id')

# outer join
df1.merge(df2,how='left',on='id')

df1.merge(df2,how='right',on='id')

df1.merge(df2,how='outer',on='id')

data = pd.merge(df1,df2,on='id',how='outer')

data = df1.join([df2,df3,df4],how='left')

# concat on row 
pd.concat([df1,df2], axis=1)

# concat on column 
pd.concat([df1,df2],ignore_index=True,axis=0)

pd.concat([df1,df2],ignore_index=True,drop_duplicates())

# combine two dataframes into one list
combine = [df1,df2]

# concat two df on column
data = df1.append(df2,ignore_index=True)

# combine two dfs with same columns
data = np.vstack((df1,df2))
data = pd.DataFrame(data)

================================================================================================

## column renaming
# set index name
data.index.name = 'name'

# rename a column in a df
data = pd.DataFrame(df,index=index).rename(columns={'0':'col1'})

# rename columns
data = data.rename(columns={'COL1':'col1', 'COL2':'col2', 'COL3':'col3'})

# assign columns name
data.columns = ['col1','col2', 'col3']

# rename columns with loop
data.columns = [x.replace('database_name','') for x in data.columns.values]

# rename columns to delete database prefix
data.columns = [i[15:] for i in data.columns]


================================================================================================

## rows/columns wrangling (including add, delete, assign value, etc)
# select one column and return a list
data['a']

data.ix[:, 'a']

# select a column and return a df
data[['a']]

# select multiple rows/columns
data.ix[0:2,0:2] # include both head and tail rows

data.iloc[0:2,0:2] # include only head but not tail rows

# delete a column
data.drop('a',axis=1,inplace=True)

# delete multiple columns
data.drop(['a','b','c'],axis=1)

# delete through del
del data['col']

# delte the column with one unique value
col_to_del = []
for i in data.columns:
	if len(data[i].value_counts) <= 1:
		col_to_del.append(i)
data = data.drop(columns=col_to_del)

# create a new column from calculation of other two cols
data['new_col'] = data['a'] - data['b']


# create columns with assign
data.assign(new_col = data['a'] - data['b'],
			new_col2 = data['a'] + data['b'])

# assign numerical value to categorical features
data['col'] = data.col.replace({'a':1, 'b':2, 'c':3})

# add a column with a mapping dictionary
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4}
data['title'] = data['title'].map(title_mapping)
data['title'] = data['title'].fillna(0)

# add a column with a mapping
data['gender'] = data['gender'].map({'female':0, 'male':1}).astype(int)

# assign value
df.score.replace(999,np.nan)

df.replace({'score':{999:np.nan}, 'name':{'Bob':np.nan}})

# assign value with slicing data
df['b'] = (df['a']==2)*1.0

# assign value through apply function
def transform(row):
	if row['group'] == 1:
		return ('class1')
	elif row['group'] == 2:
		return ('class2')

df.apply(transform,axis=1) # return the clf result
df.assign(class_n=df.apply(transform,axis=1)) # through assign function we add the clf result to a new column

# assign value through slicing data
df.loc[df.group==1,'class_n'] = 'class1'  # class_n is the new column
df.loc[df.group==2, 'class_n'] = 'class2' 


# assign value through column calculation
data['avg_price'] = data['price']/data['quantity']

# assign value with eval
data = data.eval('avg_price = price/quantity')

# assign 0 or 1 according to the comparison of other two cols
df['if_native'] = (df['total_finish_cnt_native']<=df['total_finish_cnt_web']).astype(int)

# selectively assign value with lambda 
data['class_n'] = data['group'].apply(lambda x: 'class1' if x == 1 else ('class2' if x == 2 else 'class3'))

data['fair_price'] = data['price'].apply(lambda x: 'high' if x > 300 else ('fair' if 200<x<300 else 'low'))

# assign values with np.where
df['actual_gph'] = np.where(df['group']==1,df.gph*1.5, df.gph)

df['os'] = np.where(df['a'].str.contains('windows'),'android','ios')

# assign value with cut-off
df['new_age'] = np.clip(df['age'],10,20) # all the number smaller than 10 will be re-valued at 10 and number greater than 20 will be set to 20

# assign value with function
for dataset in data:
	dataset.loc[dataset['Fare']<=7,'Fare'] = 0
	dataset.loc[(dataset['Fare']>7) & (dataset['Fare']<=14),'Fare'] = 1
	dataset.loc[(dataset['Fare']>14) & (dataset['Fare']<=31),'Fare'] = 2
	dataset.loc[dataset['Fare']>31,' Fare'] = 3
	dataset['Fare'] = dataset['Fare'].astype(int)

# find duplicates
df[df.duplicated()]

# delete duplicates
df.drop_duplicates()

df.drop_duplicates('id')

===============================================================================================================

## sorting 

# sort by certain value
df.sort_values('score',ascending=False,na_position='last')

df.sort_values(['group','score'])

# sort by index
data = data.sort_index()

# ascending with argsort
data.iloc[np.argsort(data['Age'])]

# descending with argsort
indices = np.argsort(importa)[::-1]
df = df.iloc[np.argsort(df['Age'])[::-1]]

================================================================================================

## selecting, slicing, & filter 

# select by a single condition
df[df.score>70]

# select by multiple condition
df[(df.score>70) & (df.group==1)]

# select complement group
df[~(df['group']==1)]

# select group 1 or group 2
df[(df.group==1)| (df.group==2)]

# select with query
df.query('score > 70')

df.query('price > 70 & price < 100')

df.query('(price > 70 & price < 100) | group == 1')

# select with in & index
train = df.sample(frac=0.7,random_state=1234).copy()
test = df[~df.index.isin(train.index)].copy()

# select with matching
cols = df.iloc[:,df.columns.str.startswith('is')].columns

df[df['score'].between(70,80,inclusive=True)]

df[df['name'].isin(['Bob','Lindy'])]

df[~df['name'].isin(['Bob','Lindy'])] # equals to not in

df[df['name'].str.contains('M')]

df_new = df.loc[(df['score']>70) & (df['group']==1)]

# slice numeric columns with missing data
def mis_num_cols(df):
	mis_tbl = pd.DataFrame(df.isnull().sum())
	mis_tbl = mis_tbl[mis_tbl[0] != 0].T
	mis_cols = pd.Series(mis_tbl.columns)
	all_num_cols = df.dtypes[df.dtypes!='object'].index
	mis_num_cols = [x for x in mis_cols if x in all_num_cols]
	return mis_num_cols

# slice data by dtypes
cat_cols = df.select_dtypes(['category']).columns

# slice from columns by every 2nd column
df.loc[:,'foo':'cat':2]

# slice from the beginning to 'bar'
df.loc[:,:'bar']

# slice from a column to the end by every 3 column
df.loc[:,'bar'::3]

# slice from a column to another
df.loc[:,'sat':'bar']

# slice  
df.loc[:,'sta':'bar':-1]

# slice specific columns with a list
df.loc[:,['foo','bar','dat']]

# randomly select a sub-group
ids = data.index
random_group = random.choices(ids,k=300)
temp = data.loc[random_group]

# slice a subset with index slicer
idx = pd.IndexSlice
temp = df.loc[:,idx[('col1','col2','col3'),:]]
df_final = temp[temp['col1']<=10]

# sclice data with a loop
male = [row['Sex'] for row in records if 'Sex' in row]

# slice with not-null
age = df[~df['Age'].isnull()]

# slice a list with another list
list_1 = np.arange(30)
list_2 = np.arange(60)
slice_list = [x for x in list_2 if x not in list_1]

# if index is time formating, slice with time is doable
dates.loc['11-2017']

dates.loc['11-2017':'04-2018']

usrec = DataReader('USREC','freq',start=datetime(1947,1,1),end=datetime(2013,4,1))


================================================================================================

## handling missing values

# check missing value stats
df.apply(lambda col: sum(col.isnull())/col.size)

# create a function to calculate missing values by column
def missing_values_table(df):
	mis_val = df.isnull().sum() # total missing values
	mis_val_percent = 100 * df.isnull().sum()/len(df) # percentage of missing values
	mis_val_table = pd.concat([mis_val,mis_val_percent], axis=1) # make a table storing the results
	mis_val_table_rename_cols = mis_val_table.rename(columns={0:'Missing Value', 1:'% of Total Values'}) # rename the columns

	mis_val_table_rename_cols = mis_val_table_rename_cols[mis_val_table_rename_cols[:,1]!=0].sort_values('% of Total Values', ascending=False).round(1) # sort the table by % of nulls in descending order
 	print('Your selected dataframe has' + str(df.shape[1]) + " columns.\n" 'There are '+str(mis_val_table_rename_cols.shape[0]) + ' columns that have missing values.')  # print summary 
 	return mis_val_table_rename_cols

# generate a missing table
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data

# slice data with missing % or absolute value
df1 = df.drop((missing_data[missing_data['Total']>1]).index,1)
df1.isnull().sum().max()

df2 = df.drop(missing_data[missing_data['Percent']>0.15].index,1)
df2.isnull().sum().max() # check if any null left

# delete null
df = df[df['total_traceid'].notnull()]

df = df[pd.notnull(df['CustomerId'])]

# drop rows containing missing values
df.dropna(axis=0)

# drop columns containing missing values
df.dropna(axis=1)

# only drop rows where all columns are NaN
df.dropna(how='all')

# drop rows that have less than certain(eg:4) real values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (eg: 'C')
df.dropna(subset=['C'])


# fill missing cells with specific value
df.score.fillna(df.score.mean())

df.score.fillna(df.score.median())

# fill missing value with  0 or 1
df.score.isnull().apply(int)

# generate a new category with missing values in one column
data['os_type'] = data['apr_uid'].apply(lambda x: 'old_sys' if pd.isnull(x) else 'new_sys')

# fill the nulls with another column
nulls = data[data['apt_uid'].isnull()].index
for i in nulls:
	data['apt_uid'][i] = data['second_id'][i]

# mark null as 0 and non-null as 1 in another new column ('accept')
df.loc[df['order_id'].notnull(),'accept'] = 1
df.loc[df['order_id'].isnull(),'acccept'] = 0

# fill missing values with same value('None') in multiple columns 
for col in cols:
	df[col] = df[col].fillna('None')

# fill missing values with blank
df['value'].fillna(value='blank',inplace=True)

# fill NAs with randomly assigned num within 1 std
for dataset in full_data:
	age_avg = dataset['Age'].mean()
	age_std = dataset['Age'].std()
	age_null_count = dataset['Age'].isnull().sum()
	age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
	dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
	dataset['Age'] = dataset['Age'].astype(int)

# imputing NAs with average
from sklearn.preprocessing import imputer
imr = Imputer(missing_values='NaN', strategy='mean',axis=0)  # axis = 0 means calculating column means, strategy can be set to 'median'
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

===============================================================================================================

## handling anomaly & outliers

# generate a anomaly list with threshold
def return_bad_case(df,threshold):
	q2 = df['call'].quantile(threshold)
	q1 = df['call'].quantile(1-threshold)
	df_new = df.loc[(df[call]>=q1)&(df['call']<=q2)]
	return (df_new)

df_new = return_bad_case(df,0.975)

# define anomaly with quantitle
def return_bad_case(df):
	q_lower = df['call'].quantile(0.25)
	q_upper = df['call'].quantile(0.75)
	iqu = q_upper - q_lower
	df_new = df[(df['call']>=q_lower - 1.5*iqu) & (df['call']<=q_upper+1.5*iqu)]
	return (df_new)

# fill anomaly with quantile
def cap(x,quantile=[0.01,0.99]):
	Q01, Q99 = x.quantile(quantile).values.tolist()
	if Q01 > x.min():
		x = x.copy()
		x.loc[x<Q01] = Q01

	if Q99 < x.max():
		x = x.copy()
		x.loc[x>Q99] = Q99

	return (x)

sample = pd.DataFrame({'normal':np.random.randint(1000)})
new = sample.apply(cap,quantile=[0.01, 0.99])


===============================================================================================================
## cut to bins

# fill anomaly with binsizing
pd.cut(sample.normal, bins=5, labels=[1,2,3,4,5])

pd.cut(sample.normal, bins=2, labels=['bad','good'])

pd.cut(sample.normal, bins=sample.normal.quantile([0,0.5,1]),include_lowest=True)

pd.cut(sample.normal, bins=sample.normal.quantile([0,0.5,1]),include_lowest=True,labels=['bad','good'])

df['call_time_cut'] = pd.cut(df['call_time_seconds'],
							 bins = [0,30,60,120,180,300, df['call_time_seconds'].max()],
							 precision=0, include_lowest=True)

# onehot-encoding with get dunmmies
prog_dummies = pd.ger_dummies(data['prog']).rename(columns=lambda x:'prog_'+str(x))
dataWithDummies = pd.concat([data,prog_dummies],axis=1)
dataWithDummies.drop(['prog','prog_3'],inplace=True, axis=1)
dataWithDummies = dataWithDummies.apply(np.int)
print(dataWithDummies.head())

===============================================================================================================

# 根据某个X值随机切分成A/B 两组并生成file
def cutForABtest(file='AB_test_cityallo.csv'):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file)
    df['bin']= pd.cut(df['rank'], 10,labels=range(1,11))
    df['bin'] = df['bin'].astype('int')
    res = pd.DataFrame([])
    for i in range(1,11):
        temp = df[df['bin']==i].sample(frac=0.5,replace=False)
        res = pd.concat([res, temp])
    res.reset_index(drop=True,inplace=True)
    del res['bin']
    res.to_csv('A.csv')
    df=df.append(res)
    df=df.drop_duplicates(['rank'],keep=False)
    del df['bin']
    df.reset_index(drop=True,inplace=True)
    df.to_csv('B.csv')












