
# check data types
data.info()

data.dtypes

# transform to date format
df['date'] = pd.to_datetime(df['date'])

df['date_id'] = df['date_id'].astype('datetime64')

df['time'] = pd.to_datetime(df['time'],format='%d.%m.%Y') # %d.%m.%Y=2019-01-01

# transform object to datetime
df.date = df.date.apply(lambda x: datetime.datetime.strptime(x,'%d.%m.%Y')) # %d.%m.%Y=2019-01-01

df['time'] = df['time'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x)))

# transform to category
df['order_status'] = df['order_status'].astype('category')


# convert numericals to categoricals
df['num'] = df['num'].astype(str)


# check the data types
df.dtypes.value_counts() 


# check the unique classes/values within a certain data type
df.select_dtypes('object').apply(pd.Series.nunique,axis=0)


# change the data type with loop
for col in df:
	# if column is a number with only two values, encode it as a boolean
	if (df[col].dtype != 'object') and (len(df[col].unique())<=2) # set a unique value threshold 
	df['col'] = ft.variable_types.Boolean # need to import featuretools as ft

# set a object column as category 
for col in df:
	if df[col].dtype == 'object':
		df[col] = df[col].astype('category')

# set the column as numericals
df[['col1', 'col2']] = df[['col1', 'col2']].apply(pd.to_numeric, errors='ignore')


# change the string to numericals
df['ValueNum'] = df['ValueNum'].apply(lambda x: str2number(x))

# change monetary formatting to value
def str2number(amount):
	if amount[-1] == 'M':
		return float(amount[1:-1]*1000000)
	elif amount[-1] == 'K':
		return float(amount[1:-1]*1000)
	else:
		return float(amount[1:])

# locate and convert the columns whose dtypes need to be converted
def convert_num_to_cat(df):
	num_to_cat_list = []
	numcols = df.dtypes[df.dtypes!='object'].index
	for i in numcols:
		if df[i].nunique() <= cat_threshold_maxnum # cat_threshold_maxnum is a preset threshold
		num_to_cat_list.append(i)
		df[num_to_cat_list] = df[num_to_cat_list].astype(str)


===============================================================================================================

## Time Series Change
# change other formating to ts
data['alert_time'] = pd.to_datetime(data['alert_time'],unit='s')

train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')   #%d.%m.%Y = 2019-03-06

sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y')) # chaneg obj to ts

data['altert_time'] = data['tmp0429.alert_time'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))

# create ts series

longer_ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000',periods=1000)) # 创建1个间隔为天的ts range

dates = pd.date_range('1/1/2000', periods=10, freq=‘M’) # 创建1个间隔为月的ts range

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000','1/2/2000', '1/3/2000'])

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000','1/2/2000', '1/3/2000'])  # 创建1个间隔为季度的ts range

pd.date_range('2000-01-01', periods=10, freq='1h30min') # 创建1个间隔为1.5小时的ts range


# extract ts element
train['month'] = train['date'].dt.month   # extract month from a date

train['day'] = train['date'].dt.day

train['weekday'] = train['date'].dt.weekday 

train['weekday'] = train['date'].dt.year

# 提取hour/minute/second
data['time'].dt.hour
data['time'].dt.minute
data['time'].dt.second











