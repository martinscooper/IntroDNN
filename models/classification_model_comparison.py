from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utilfunction import *
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
seed = 7
import sklearn
numeric_cols = ['cantConversation', 'wifiChanges',
                'stationaryCount', 'walkingCount', 'runningCount', 'silenceCount', 'voiceCount', 'noiseCount',
                'unknownAudioCount']

transformer = ColumnTransformer([('transformer', StandardScaler(),
                                  numeric_cols)],
                                remainder='passthrough')

clf = LogisticRegression(random_state=seed, solver='lbfgs',
                         multi_class='ovr', max_iter=400)

model = make_pipeline(transformer, clf)

df = pd.read_pickle('sedentarism3.pkl')

precision1, recall1 = per_user_classification(df, model)
precision2, recall2 = live_one_out_classification(df, model)

plt.close()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(precision1, label='per_user_classification')
ax1.plot(precision2, label='live_one_out_classification')

ax2.plot(recall1)
ax2.plot(recall2)

fig.legend()
fig.show()
