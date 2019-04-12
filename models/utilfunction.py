from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression

def createSensingTable(path):
    df = pd.read_csv(path + '00' + '.csv', index_col=False)
    df['userId'] = '00'
    for a in range(1, 60):
        userId = '0' + str(a) if a < 10 else str(a)
        try:
            aux = pd.read_csv(path + userId + '.csv', index_col=False)
            aux['userId'] = a
            df = df.append(aux)
        except:
            pass
    return df


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def get_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) == userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

def get_non_user_data(data, userId):
    try:
        return data.loc[data.index.get_level_values(0) != userId].copy()
    except KeyError:
        print('El usuario ', userId, ' no existe.')

def gen_live_one_out(df):
    for i in df.index.get_level_values(0).drop_duplicates():
        yield get_non_user_data(df,i), get_user_data(df,i)

def get_X_y_regression(df):
    features = [col for col in df.columns if 'isSedentary' != col]
    return df[features].copy(), df['isSedentary'].copy()

def get_X_y_classification(df):
    df['sclass'] = ''
    df.loc[df['isSedentary'] > 0.9999, 'sclass'] = 0  # 'very sedentary'
    df.loc[df['isSedentary'].between(0.9052, 0.9999), 'sclass'] = 1  # 'sedentary'
    df.loc[df['isSedentary'] < 0.9052, 'sclass'] = 2  # 'less sedentary'
    features = [col for col in df.columns if 'sclass' != col]
    return df[features].copy(), df['sclass'].copy()

def per_user_regression(df, model):
    print('per_user_regression')
    seed = 7
    mse = []
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_regression(get_user_data(df, userid))
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        mse.append(-results.mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return mse

def live_one_out_regression(df, model):
    print('live_one_out_regression')
    seed = 7
    mse = []
    i = 0
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_regression(df)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return mse


def per_user_classification(df, model):
    print('per_user_classification')
    scoring = ['precision_weighted', 'recall_weighted']
    seed = 7
    precision = []
    recall = []
    for userid in df.index.get_level_values(0).drop_duplicates():
        X, y = get_X_y_classification(get_user_data(df, userid))
        kfold = KFold(n_splits=10, random_state=seed)
        results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        precision.append(results['test_precision_weighted'].mean())
        recall.append(results['test_recall_weighted'].mean())
        if userid % 10 == 0:
            print('modelos sobre usuario ', userid, ' finalizado.')
    return precision, recall

def live_one_out_classification(df, model):
    print('live_one_out_classification')
    i = 0
    precision = []
    recall = []
    logo = LeaveOneGroupOut()
    groups = df.index.get_level_values(0)
    X, y = get_X_y_classification(df)
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        if i % 10 == 0:
            print('modelos sobre usuario ', i, ' finalizado.')
        i += 1
    return precision, recall

def shift_hours(df):
    return df.sort_index()['isSedentary'].shift(-1)