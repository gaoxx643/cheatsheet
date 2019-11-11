## grouping calculation

# simple calculation through grouping
sample.groupby('grade')[['math']].max()

sample.groupby(['grade','class'])[['math']].mean()

sample.groupby(['grade'])['math','chinese'].mean()

sample.groupby('class')['math'].agg(['mean','min','max'])

df = sample.groupby(['grade','class'])['math','chinese'].agg(['mean','max'])

df.groupby('scene').agg({'accepted':'sum','date':'count'})

df.groupby('type')['swing','month_freq'].agg({'mean':'mean','median':'median','cnt':'count'})

# calculate rate through lambda function
df['rate'] = df.groupby(['order_stats'])['cnt'].apply(lambda x: x/sum(x))

===============================================================================================================

## pivoting table (long to wide table)
pd.pivot_table(df, index='id', columns='type', values='monetary')

pd.pivot_table(df, index='id', columns='type', values='monetary', fill_value=0, aggfunc='sum')

df_summary = pd.pivot_table(index='show_times',columns='accept_flag', values=['total_accept','show_volume'],aggfunc={'total_accept':'sum','show_volume':'count'})

store_stats = pd.pivot_table(index='shop_id', columns='item_category_id', values='item_cnt_month',aggfunc='count',fill_value=0)

df_summary = df.pivot_table(index=['dates','city_id'], columns='experiment_group',values=['gmv','finish_order'],aggfunc='sum')

===============================================================================================================

## reshape table with unstack - similar to pivot_table(long to wide table) 
# aggregated results can be reshaped into a table with unstack
agg_counts = df.unstack().fillna(0)

monthly_sales = sales.groupby(['shop_id','item_id','date_block_num'])['item_cnt_day'].sum()
monthly_sales = monthly_sales.unstack(level=-1).fillna(0)
monthly_sales = monthly_sales.T
dates = pd.date_range(start='2013-01-01', end='2015-10-01', freq='MS')
monthly_sales.index = dates
monthly_sales = monthly_sales.reset_index()


## change wide table to long table
# change wide table to long table with melt
pd.melt(df, id_vars='id', value_vars=['Normal','Special_offer'],value_name='Monetary',var_name='Type')

# change wide table to long table with stack
count_subset = count_subset.stack() 


