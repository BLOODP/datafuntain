import gc
import time

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

print('.................')
farm_reader = pd.read_csv('/workplace/DF/FutureData/farming.csv', iterator=True)

market = pd.read_csv('/workplace/DF/FutureData/product_market.csv', parse_dates=['数据入库时间', '数据发布时间'])
market.drop('规格',axis=1,inplace=True)
market.drop('颜色',axis=1,inplace=True)
market.drop('单位',axis=1,inplace=True)
market.drop('区域',axis=1,inplace=True)
market.columns = ['provence', 'market_name', 'category', 'name', 'database_time', 'public_time']
market['month_of_year'] = market.public_time.dt.month
market['day_of_week'] = market.public_time.dt.dayofweek
market['day_of_month'] = market.public_time.dt.day
market['day_of_year'] = market.public_time.dt.dayofyear

market.drop('database_time', axis=1, inplace=True)
market = market.dropna()

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
del chunks
gc.collect()
print(farm.shape)
print()

farm['month_of_year'] = farm.public_time.dt.month
farm['day_of_week'] = farm.public_time.dt.dayofweek
farm['day_of_month'] = farm.public_time.dt.day
farm['day_of_year'] = farm.public_time.dt.dayofyear

farm.columns = ['provence', 'market_name', 'category', 'name', 'averaging', 'database_time', 'public_time',
                'month_of_year', 'day_of_week', 'day_of_month', 'day_of_year']

farm.drop('database_time', axis=1, inplace=True)
farm = farm.dropna()


def striped(s):
    s = s[:4] + s[-4:]
    return s


print('striping ............ ')

unique_names = farm['name'].unique()
print(len(unique_names))


def cal_days(d):
    g = pd.to_datetime(str(d)) - pd.to_datetime(str(sorted_pub.values[0]))
    return g.days


def season(m):
    if m in [3, 4, 5]:
        s = 0
    elif m in [6, 7, 8]:
        s = 1
    elif m in [9, 10, 11]:
        s = 2
    else:
        s = 3
    return s

error_sum = 0.0
strat = time.time()

submit = None
flag = False
i = 0
print('###############################################')

for name in unique_names[:500]:
    df_train = farm[farm['name'] == name]
    df_train = df_train.sort_values(by='public_time')

    df_test = market[market['name'] == name]
    df_test = df_test.sort_values(by='public_time')

    i += 1
    if df_test.shape[0] == 0:
        print('test shape 0 ')
        continue

    sorted_pub = df_train.public_time.sort_values()

    df_train['days'] = df_train.public_time.apply(lambda x: cal_days(x))
    df_test['days'] = df_test.public_time.apply(lambda x: cal_days(x))

    Y_train = df_train['averaging'].copy()
    # df_train.drop('averaging', axis=1, inplace=True)

    df = pd.concat([df_train,df_test])
    df.drop('public_time',axis=1,inplace=True)

    df['market_mean'] = pd.Series([], dtype='float64')
    # df['market_std'] = pd.Series([], dtype='float64')
    df['market_min'] = pd.Series([], dtype='float64')
    df['market_max'] = pd.Series([], dtype='float64')

    groups = df.groupby('market_name')
    for n, group in groups:
        it = group.shape[0]
        if it > 0:
            market_mean = group[group['averaging'].notnull()]['averaging'].mean()
            # market_std = group[group['averaging'].notnull()]['averaging'].std()
            market_min = group[group['averaging'].notnull()]['averaging'].min()
            market_max = group[group['averaging'].notnull()]['averaging'].max()
            df = df.set_value(group.index, 'market_mean', round(market_mean, 2))
            # df = df.set_value(group.index, 'market_std', round(market_std, 2))
            df = df.set_value(group.index, 'market_min', round(market_min, 2))
            df = df.set_value(group.index, 'market_max', round(market_max, 2))
    # df.fillna(0,inplace=True)
    df.drop('averaging', axis=1, inplace=True)

    print('dumming ..... ')
    month_of_year_dummies = pd.get_dummies(df['month_of_year'], prefix='month')
    market_name_dummies = pd.get_dummies(df['market_name'], prefix='market_name')
    weekday_dummies = pd.get_dummies(df['day_of_week'], prefix='weekday')
    mothday_dummies = pd.get_dummies(df['day_of_month'], prefix='month_day')

    df = df.drop(['name'], axis=1)
    df = df.drop(['day_of_week'], axis=1)
    df = df.drop(['day_of_month'], axis=1)
    df = df.drop(['day_of_year'], axis=1)
    df = df.drop(['category'], axis=1)
    df = df.drop(['market_name'], axis=1)
    df = df.drop(['provence'], axis=1)
    df = df.drop(['month_of_year'], axis=1)

    print('concating.....')
    df = pd.concat([df, month_of_year_dummies], axis=1)
    df = pd.concat([df, market_name_dummies], axis=1)
    df = pd.concat([df, weekday_dummies], axis=1)
    df = pd.concat([df,mothday_dummies],axis=1)

    rows = df_train.shape[0]

    X_train = df.iloc[:rows]

    X_val = df.iloc[rows:]

    print(X_train.shape)
    print(X_val.shape)
    print()

    print('training ......')

    est = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=0, loss='ls',max_features='sqrt')
    est.fit(X_train, Y_train)

    print('predicting ..........')
    pred = est.predict(X_val.iloc[:])
    sub = pd.Series(pred,index=X_val.index,name='predicted')
    sub = pd.concat([df_test.iloc[:][['market_name','name','public_time']],sub],axis=1)
    if flag:
        submit = pd.concat([submit,sub])
    else:
        submit = sub
        flag = True

    print("---------------------  ",i,"   -----------------")
    print()
    print()
end = time.time()
t = end - strat
print('time : ', t)
submit.to_csv('/workplace/submit5_0_500_.csv',index=False)