import pickle
from sklearn.metrics import mean_squared_error

def test_error(user, architecture):
    info = cache[user][architecture]
    x_test, y_test, model = info['x_test'], info['y_test'], info['model']
    y_pred = model.predict(x_test)
    return round(mean_squared_error(y_test, y_pred), 3)

def test_all():
    print('')
    print('*' * 16)
    for i in users:
        for j in range(number_of_architectures):
            mse = test_error(i, j + 1)
            print('El mse para el usuario {0} utilizando la arquitectura {1} fue: {2}'.format(i, j + 1, mse))

    print('*' * 16)
    print('')

users = [50, 31, 4]
number_of_architectures = 6


cache = pickle.load(open('testing_info.pkl', 'rb'))
test_all()