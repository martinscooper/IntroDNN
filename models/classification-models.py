from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from utilfunction import *
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_pickle('sedentarism4.pkl')
X, y = get_X_y_classification(df)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

numeric_cols = ['cantConversation', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount']
ss = StandardScaler()
X_train.loc[:, numeric_cols] = ss.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = ss.transform(X_test[numeric_cols])
'''
#codigo para usar oversampling
columns = X.columns
sm = SMOTE(random_state=12, ratio='all')
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_train, columns=columns)
y_train = pd.Series(y_train)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))
'''
size = X.shape[1]
# Initialize the constructor
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(size,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['categorical_accuracy'])

h = model.fit(X_train, y_train, epochs=20, batch_size=128, verbose=2,
          validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

#model.summary()
