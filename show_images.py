import sys
import pickle
import matplotlib.pyplot as plt

def show_train_prediction(user, architecture):
    info = cache[user][architecture]
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Train data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = info['model'].predict(info['x_train'])
    plt.plot(info['y_train'], label='Train')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_test_prediction(user, architecture):
    info = cache[user][architecture]
    plt.close()
    plt.figure(figsize=(15, 4))
    plt.title('Test data of user {0} with architecture {1}'.format(user, architecture))
    y_pred = info['model'].predict(info['x_test'])
    plt.plot(info['y_test'], label='Test')
    plt.plot(y_pred, label='Predicted')
    plt.axhline(y=1.5, color='r', linestyle=':', )
    plt.legend(loc='upper right')
    plt.show()


def show_history_loss(user, architecture):
    history = cache[user][architecture]['history']
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


if __name__ == "__main__":
    cache = pickle.load(open('testing_info.pkl', 'rb'))
    show_images(sys.argv[1], sys.argv[2])

