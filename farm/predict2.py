import gc
import time

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#  读取CSV文件
farm_reader = pd.read_csv('/workplace/DF/FutureData/farming.csv', iterator=True)
market = pd.read_csv('/workplace/DF/FutureData/product_market.csv', parse_dates=['数据入库时间', '数据发布时间'])

loop = True
chunks = []

# 由于数据集文件较大，无法一次性读入至内存，因此迭代读入并做数据转换
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
farm.dropna()

farm['month_of_year'] = farm.public_time.dt.month
farm['day_od_week'] = farm.public_time.dt.dayofweek
farm['day_od_month'] = farm.public_time.dt.day
farm['day_od_year'] = farm.public_time.dt.dayofyear

farm.columns = ['provence', 'market_name', 'category', 'name', 'averaging', 'database_time', 'public_time',
                'month_of_year', 'day_od_week', 'day_of_month', 'day_od_year']

farm.drop('database_time', axis=1, inplace=True)


def striped(s):
    s = s[:4] + s[-4:]
    return s


print('striping ............ ')
farm['market_name'] = farm['market_name'].apply(lambda x: striped(x))
farm['name'] = farm['name'].apply(lambda x: striped(x))

unique_names = farm['name'].unique()
print(len(unique_names))


def cal_days(d):
    g = pd.to_datetime(str(d)) - pd.to_datetime(str(sorted_pub.values[0]))
    return g.days


# 特征工程
error_sum = 0.0
strat = time.time()
for name in unique_names[:10]:
    df = farm[farm['name'] == name]
    if df.shape[0] < 10:
        continue
    df = df.sort_values(by='public_time')
    sorted_pub = df.public_time.sort_values()

    df['days'] = df.public_time.apply(lambda x: cal_days(x))

    farm['market_mean'] = pd.Series([], dtype='float64')
    farm['market_median'] = pd.Series([], dtype='float64')
    farm['market_min'] = pd.Series([], dtype='float64')
    farm['market_max'] = pd.Series([], dtype='float64')

    groups = farm.groupby('market_name')
    for n, group in groups:
        it = group.shape[0]
        if it > 0:
            market_mean = group.iloc[:int(round(0.8 * it))]['averaging'].mean()
            market_median = group.iloc[:int(round(0.8 * it))]['averaging'].median()
            market_min = group.iloc[:int(round(0.8 * it))]['averaging'].min()
            market_max = group.iloc[:int(round(0.8 * it))]['averaging'].max()
            farm = farm.set_value(group.index, 'market_mean', round(market_mean, 2))
            farm = farm.set_value(group.index, 'market_median', round(market_median, 2))
            farm = farm.set_value(group.index, 'market_min', round(market_min, 2))
            farm = farm.set_value(group.index, 'market_max', round(market_max, 2))

    print('dumming ..... ')
    month_of_year_dummies = pd.get_dummies(df['month_of_year'], prefix='month')
    df = df.drop(['provence'], axis=1)
    df = df.drop(['month_of_year'], axis=1)
    market_name_dummies = pd.get_dummies(df['market_name'], prefix='market_name')
    df = df.drop(['market_name'], axis=1)
    df = df.drop(['category'], axis=1)
    weekday_dummies = pd.get_dummies(df['day_od_week'], prefix='weekday')
    mothday_dummies = pd.get_dummies(df['day_of_month'], prefix='month_day')

    df = df.drop(['name'], axis=1)
    df = df.drop(['day_od_week'], axis=1)
    df = df.drop(['day_of_month'], axis=1)
    df = df.drop(['day_od_year'], axis=1)

    df = pd.concat([df, month_of_year_dummies], axis=1)
    df = pd.concat([df, market_name_dummies], axis=1)
    df = pd.concat([df, weekday_dummies], axis=1)
    df = pd.concat([df, mothday_dummies], axis=1)

    rows = df.shape[0]

    X_train = df.iloc[:int(0.8 * rows)]
    X_train.to_csv('/workplace/X_train.csv', index=False)
    Y_train = X_train['averaging'].copy()
    X_train = X_train.drop('averaging', axis=1)
    # X_normalized = preprocessing.normalize(X_train, norm='l2')
    X_train = X_train.drop('public_time', axis=1)

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

    est = GradientBoostingRegressor(n_estimators=140, learning_rate=0.1, max_depth=4, random_state=0,
                                    loss='ls', max_features='sqrt')
    est.fit(X_train, Y_train)
    r = est.predict(X_val.iloc[:20])
    pred = est.predict(X_val.iloc[:])

    print(r)
    print(Y_val.iloc[:20])
    error = mean_squared_error(Y_val.iloc[:], pred)
    error_sum += error
    print('error : ', error)
    print("--------------------------------------")
    print()
    print()
end = time.time()
t = end - strat
print('time : ', t)
print('error sum : ', error_sum)
