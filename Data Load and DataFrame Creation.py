
## load a single file

# load csv file
data = pd.read_csv('filename', sep='\t', header=None)

data = pd.read_csv('filename',parse_dates=True,index_col=0)

# load txt file
f = open('filename.txt', 'r').read()

# read excel file
file = '/Desktop/DS/filename.xlsx'
data = pd.ExcelFile(file)
data = xlsx.parse('tab_name')

# load json file with loop
path = 'Desktop/filename.txt'
open(path).readline()
import json
records = [json.loads(line) for line in open(path)]

# load json file with json
data = json.load(open('Desktop/filename.json'))

# load online file
data = pd.read_stata('http://www.stata-press.com/data/r14/rgnp.dta').iloc[1:]

filename = requests.get('http://econ.korea.ac.kr/~cjkim/MARKOV/data/ew_excs.prn').content
data = pd.read_table(BytesIO(filename),header=None, skipfooter=1, engine='python')

url =  'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'
data = pd.read_csv(url)


## load a big-size file 
# load a big single file with chunks
traintypes = {
			  'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'
              } # set columns to most suitable type to optimize for memory usage

cols = list(traintypes.keys())
chunksize = 5_000_000 # 5 million rows at onece. 

%%time
df_list = [] # create a list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
	 # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
     # Using parse_dates would be much slower!
     df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0,16)
     df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'],utc=True,format='%Y-%m-%d %H:%M')  
     df_list.append(df_chunk) # Alternatively, append the chunk to list and merge all


# load a big file with loop
import csv
import numpy as np
with open('Desktop/filename.csv',, 'rt') as csvfile:
	file_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
	row = next(file_reader) # header contains feature names
	feature_names = np.array(row)

# load dataset and target classes
titanic_X, titanic_y = [], []
for row in file_reader:
	titanic_X.append(row)
	titanic_y.append(row[2]) # the target value is 'survived'
	titanic_X = np.array(titanic_X)
	titanic_y = np.array(titanic_y)


## load data from multiple files within one 
# method 1
data_dir = 'your_directory'  # set the directory storing data

data = pd.DataFrame()


for file in os.listdir(data_dir):
	dataset = pd.read_csv(os.path.join(data_dir,file), sep='\t', header=None)
	data = data.append(dataset)

data.columns = ['column1', 'column2', 'column3', 'column4']


# method 2
years = range(1880,2011) 
pieces = []
columns = ['name','sex','births'] # setup columns names
for year in years:
	path = 'Desktop/directory/yob%d.txt' % year # pipe in year into the file name
	frame = pd.read_csv(path,names=columns)
	frame['year'] = year
	pieces.append(frame)

df = pd.concat(pieces,ignore_index=True)


## to create a dataframe 
# create a df with df
temp = {'id':[],'value':[]}
for i in range(1000):
	cal = perm_func(df,139,53)
	temp['id'].append(i)
	temp['value'].append(cal)

df = pd.DataFrame.from_dict(temp)

# create a df within a range
df = pd.DataFrame(np.arange(20).reshape(4,5),columns=list('abcde'))

# create a df through extrating cols from anotehr df
df_temp = pd.DataFrame({'hour':df['hour'],
						'city_id':df['city_id'],
						'orders':df['orders'],
						'recommend_orders':df['recommend_orders']})







