import sys
import pickle
import matplotlib.pyplot as plt


def show_train_prediction(user, architecture):
    info = train_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Train data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = model.predict(info['x_train'])
    plt.plot(info['y_train'], label='Train')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_test_prediction(user, architecture):
    info = test_cache[user][architecture]
    model = models[user][architecture]['model']
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Test data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = model.predict(info['x_test'])
    plt.plot(info['y_test'], label='Test')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_history_loss(user, architecture):
    history = models[user][architecture]['history']
    plt.close()
    plt.title('Train loss vs. Test loss of user {0} with architecture {1}'.format(user, architecture))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def show_images(user, architecture):
    # info = pickle.load(open('testing_info.pkl', 'rb'))
    show_train_prediction(user, architecture)
    show_test_prediction(user, architecture)
    show_history_loss(user, architecture)


print('*** Generando predicciones para el usuario {0} y la arquitectura {1}... ***'.format(sys.argv[1], sys.argv[2]))
train_cache = pickle.load(open('train_cache.pkl', 'rb'))
test_cache = pickle.load(open('test_cache.pkl', 'rb'))
models = pickle.load(open('models.pkl', 'rb'))
show_images(int(sys.argv[1]), int(sys.argv[2]))

