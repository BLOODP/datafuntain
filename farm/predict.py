import gc

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print(".................")
farm_reader = pd.read_csv('/workplace/DF/FutureData/farming.csv', iterator=True)

loop = True
chunks = []

while loop:
    try:
        chunk = farm_reader.get_chunk(100000)
        chunk['database_time'] = pd.to_datetime(chunk['数据入库时间'], errors='coerce')
        chunk['public_time'] = pd.to_datetime(chunk['数据发布时间'], errors='coerce')
        chunk.drop('规格', axis=1, inplace=True)
        chunk.drop('颜色', axis=1, inplace=True)
        chunk.drop('单位', axis=1, inplace=True)
        chunk.drop('区域', axis=1, inplace=True)
        chunk.drop('数据发布时间', axis=1, inplace=True)
        chunk.drop('数据入库时间', axis=1, inplace=True)
        chunk.drop('最低交易价格', axis=1, inplace=True)
        chunk.drop('最高交易价格', axis=1, inplace=True)
        print('shape : ', chunk.shape)
        chunks.append(chunk)
        del chunk
        gc.collect()
    except StopIteration:
        loop = False

farm = pd.concat(chunks, ignore_index=True)
farm = farm.dropna()
del chunks
gc.collect()
print(farm.shape)
print()

farm['month_of_year'] = farm.public_time.dt.month
farm['day_od_week'] = farm.public_time.dt.dayofweek
farm['day_od_month'] = farm.public_time.dt.day
farm['day_od_year'] = farm.public_time.dt.dayofyear


farm.columns = ['provence', 'market_name', 'category', 'name', 'averaging', 'database_time', 'public_time',
                'month_of_year', 'day_od_week', 'day_of_month', 'day_od_year']
# print(farm.columns)
farm.drop('database_time', axis=1, inplace=True)


def striped(s):
    s = s[:4] + s[-4:]
    return s


print('striping ............ ')
farm['market_name'] = farm['market_name'].apply(lambda x: striped(x))

unique_names = farm['name'].unique()
print(len(unique_names))

farm = farm[farm['name'] == '10897751E8AA8258910F731B34488E5C']
farm = farm.sort_values(by='public_time')
# farm['days_shift'] = pd.Series([])

sorted_pub = farm.public_time.sort_values()


def cal_days(d):
    g = pd.to_datetime(str(d)) - pd.to_datetime(str(sorted_pub.values[0]))
    return g.days


farm['days'] = farm.public_time.apply(lambda x: cal_days(x))
farm['market_mean'] = pd.Series([],dtype='float64')
farm['market_std'] = pd.Series([],dtype='float64')
farm['market_min'] = pd.Series([],dtype='float64')
farm['market_max'] = pd.Series([],dtype='float64')
# farm['days'] = pd.Series([])


#
groups = farm.groupby('market_name')
for n,group in groups:
    it = group.shape[0]
    if it > 0:
        # shift = group.public_time - group.public_time.shift(1)
        # shift.fillna(0,inplace=True)
        # shift = shift.apply(lambda x: x.days)
        # farm = farm.set_value(shift.index,'days_shift',shift.iloc[:])
        # sorted_pub = group.public_time.sort_values()
        # farm = farm.set_value(group.index,'days',group.public_time.apply(lambda x:cal_days(x)))
        market_mean = group.iloc[:int(round(0.8*it))]['averaging'].mean()
        market_std = group.iloc[:int(round(0.8*it))]['averaging'].std()
        market_min = group.iloc[:int(round(0.8 * it))]['averaging'].min()
        market_max = group.iloc[:int(round(0.8 * it))]['averaging'].max()
        farm = farm.set_value(group.index,'market_mean',round(market_mean,2))
        farm = farm.set_value(group.index, 'market_std', round(market_std, 2))
        farm = farm.set_value(group.index, 'market_min', round(market_min, 2))
        farm = farm.set_value(group.index, 'market_max', round(market_max, 2))
farm.market_std.fillna(0,inplace=True)

print('dumming ..... ')
month_of_year_dummies = pd.get_dummies(farm['month_of_year'], prefix='month')
farm = farm.drop(['provence'], axis=1)
farm = farm.drop(['month_of_year'], axis=1)
market_name_dummies = pd.get_dummies(farm['market_name'], prefix='market_name')
farm = farm.drop(['market_name'], axis=1)
farm = farm.drop(['category'], axis=1)
weekday_dummies = pd.get_dummies(farm['day_od_week'], prefix='weekday')
mothday_dummies = pd.get_dummies(farm['day_of_month'], prefix='month_day')
farm = farm.drop(['name'], axis=1)
farm = farm.drop(['day_od_week'], axis=1)
farm = farm.drop(['day_of_month'], axis=1)
farm = farm.drop(['day_od_year'], axis=1)

df = pd.concat([farm, month_of_year_dummies], axis=1)
df = pd.concat([df, market_name_dummies], axis=1)
df = pd.concat([df, weekday_dummies], axis=1)
df = pd.concat([df, mothday_dummies], axis=1)

print(df.shape)
print(df.head(2))

rows = df.shape[0]

# X_train = df[(df.public_time < pd.to_datetime('2015-11-01'))]
X_train = df.iloc[:int(0.8 * rows)]
X_train = X_train
X_train.to_csv('/workplace/X_train.csv', index=False)
Y_train = X_train['averaging'].copy()
X_train = X_train.drop('averaging', axis=1)
# X_normalized = preprocessing.normalize(X_train, norm='l2')
X_train = X_train.drop('public_time', axis=1)

# X_val = df[(df.public_time >= pd.to_datetime('2015-11-01'))]
X_val = df.iloc[int(0.8 * rows):]
X_val.to_csv('/workplace/X_val.csv', index=False)
Y_val = X_val['averaging'].copy()
X_val = X_val.drop('averaging', axis=1)
X_val = X_val.drop('public_time', axis=1)

print(X_train.shape)
print(X_val.shape)

print(X_train.dtypes)
print(Y_train.dtypes)

print()

print('training ......')

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=0,
                                loss='ls',max_features='sqrt')
est.fit(X_train, Y_train)

print('predicting..........')
r = est.predict(X_val.iloc[:20])
pred = est.predict(X_val.iloc[:])

print(r)
print(Y_val.iloc[:20])
print('error : ', mean_squared_error(Y_val.iloc[:], pred))

# param_test1 = {'max_depth':list(range(4,10))}
# gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,
#                                                             max_features='sqrt',subsample=0.8,
#                                                             random_state=0),
#                         param_grid=param_test1, cv=5,scoring='mean_squared_error')
# gsearch1.fit(X_train,Y_train)
# print(gsearch1.grid_scores_)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)
# pred = gsearch1.best_estimator_.predict(X_val.iloc[:])
# print('error : ', mean_squared_error(Y_val.iloc[:], pred))
