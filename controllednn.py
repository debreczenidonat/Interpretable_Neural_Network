from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import seaborn as sbn


#from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPool1D, Reshape, UpSampling1D,BatchNormalization, multiply, add, Input, MaxPooling1D, LocallyConnected1D, AveragePooling1D, TimeDistributed, RepeatVector, CuDNNLSTM, Dot, dot
from keras.layers import Flatten
from keras.utils import normalize
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import hashing_trick, Tokenizer
#from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras import regularizers
from keras.initializers import RandomUniform
from keras.initializers import Orthogonal
from keras.initializers import RandomNormal
from keras.initializers import Zeros
from sklearn import preprocessing
from keras.constraints import UnitNorm, MinMaxNorm, NonNeg
from keras.regularizers import l1_l2

K.tensorflow_backend._get_available_gpus()




###READ IN DATA
tt = pd.read_csv('titanic/train.csv')
tt.columns
tt = tt[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',  'Parch', 'Fare', 'Embarked']]
tt = tt.dropna()
tt['IsChild'] = tt['Age'] <= 10
tt['IsChild'] = tt['IsChild'].apply(lambda x: str(x))
tt['codedcats'] = tt['Pclass'].apply(lambda x: str(x)) + tt['Sex'] + tt['Embarked'] + tt['IsChild']
LEN_OF_CATS = len(tt['codedcats'].unique())

tok = Tokenizer(LEN_OF_CATS, lower=False)#, oov_token='Other'
tok.fit_on_texts(tt['codedcats'])

tt['codedcats'] = tok.texts_to_sequences(tt['codedcats'])
tt['codedcats'] = tt['codedcats'].apply(lambda x: np.squeeze(x).astype(float))
tt['codedcats'] = pd.to_numeric(tt['codedcats'])



###NAME DATASETS
ttcon = tt[['Age', 'SibSp', 'Parch', 'Fare']]
ttd = tt[['Sex', 'Pclass', 'Embarked', 'IsChild']]
# ttdisc = tt[['female','male', '1', '2',  '3', 'C', 'Q',  'S']]
# ttcoded = ttdisc['female'].astype(str) + ttdisc['male'].astype(str) + ttdisc['1'].astype(str) + ttdisc['2'].astype(str) + ttdisc['3'].astype(str) + ttdisc['C'].astype(str) + ttdisc['Q'].astype(str) + ttdisc['S'].astype(str)
ttcoded = tt['codedcats']
tttarget = tt[['Survived']]

summaries = tt.groupby(by=['codedcats','Sex','Embarked','Pclass','IsChild']).count().reset_index()
summaries = summaries[['codedcats','Sex','Embarked','Pclass', 'IsChild', 'Survived']]
summaries.columns = ['codedcats','Sex','Embarked','Pclass', 'IsChild', 'Count']


###CUSTOM REG FUNCS
def custom_activity_reg(vects, a=1.0):
    y = K.l2_normalize(vects)
    #return a * (1.0 - K.sum(y * y))
    return a * (1.0 - K.sum(y * y * y * y))


def custom_activity_regII(vects):
    y = vects - K.mean(K.identity(vects), keepdims=True, axis=-1)
    s = K.cast(K.shape(vects)[-1], 'float32')
    fourth_momentum = K.sum(y * y * y * y) / s
    second_momentum = K.sum(y * y) / s
    return (fourth_momentum / (second_momentum * second_momentum))-3


def custom_activity_regIII(vects):
    return 1 / K.abs(K.mean(K.identity(vects)) - K.max(K.identity(vects)))


def custom_activity_regIV(vects):
    a = K.abs(K.mean(K.identity(vects)) / K.max(K.identity(vects)))
    b = K.abs(K.sum(K.identity(vects)) / K.max(K.identity(vects)))
    c = K.abs(K.max(K.identity(vects)) - 1)
    d = K.abs(K.sum(K.identity(vects)) - 1)
    return 0.2*(K.log(b) + K.log(1+c))



###BUILD MODEL

NUMBER_OF_LIN_MODS = 5

input_layer_cont = Input(shape=[4])
bn1 = BatchNormalization(center=False, scale=False)(input_layer_cont)
input_layer_disc = Input(shape=[1])
emb1 = Embedding(LEN_OF_CATS, NUMBER_OF_LIN_MODS, embeddings_constraint=NonNeg(), activity_regularizer=custom_activity_regIV, embeddings_initializer=Orthogonal())(input_layer_disc)#,
flat1 = Flatten()(emb1)
#densea = Dense(3, activation='sigmoid', use_bias=False, kernel_constraint=, kernel_initializer=Zeros())(flat1)#kernel_initializer=Orthogonal(gain=1),kernel_regularizer=regularizers.l2(0.9)
denseb = Dense(NUMBER_OF_LIN_MODS, activation='sigmoid',use_bias=True)(bn1)
dense2 = dot([emb1, denseb], axes=-1)
#dense2 = Dense(1, activation='linear', bias_initializer=RandomUniform(minval=15, maxval=25, seed=None))(dotpr)#, kernel_initializer=Orthogonal(gain=1.0, seed=None),

model = Model(inputs=[input_layer_cont, input_layer_disc], outputs=[dense2])
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
model.summary()

####
# model.layers[3].get_weights()
# model.layers[3].set_weights([np.array([
#        [10., -10., -10., -10.],
#        [-10., 10., -10., -10.],
#        [-10., -10., 10., -10.],
#        [-10., -10., -10., 10.],
#        [10., -10., 10., -10.]], dtype='float32')]
#                             )
#
# model.layers[3].set_weights([np.array([
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]], dtype='float32')]
#                             )

#TRAIN MODEL
early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=600, verbose=0, mode='auto')#monitor='val_loss'
h = model.fit([ttcon, ttcoded], tttarget, epochs=8000, verbose=1, batch_size=2000, callbacks=[early_stopping])#, validation_data=(ts_test_features, ts_test_target), callbacks=[early_stopping]

model.get_config()
for l in model.layers:
    print(l.get_weights())

#model.evaluate([ttcon, ttcoded], tttarget)


test = [ttcon.values.astype('float32'), ttcoded.values.astype('float32')]

get_disc_layer_output = K.function([*model.input], [model.layers[3].output])
get_cont_layer_output = K.function([*model.input], [model.layers[4].output])
get_norm_layer_output = K.function([*model.input], [model.layers[2].output])
layer_disc_output = get_disc_layer_output(test)[0]
layer_cont_output = get_cont_layer_output(test)[0]
layer_norm_output = get_norm_layer_output(test)[0]

layer_disc_output_u = np.unique(layer_disc_output, axis=0)


results = np.empty(shape=(0))
for i in range(layer_disc_output.shape[0]):
    results = np.append(results, (np.dot(layer_disc_output[i], layer_cont_output[i])))


#results = 1/(1+np.exp(-1*results))
preds = model.predict(test)
print(preds[5])
print(results[5])

model.layers[4].get_weights()
# print(layer_disc_output)
# print(layer_cont_output)
# np.dot(layer_disc_output[0], layer_cont_output[0])
# model.predict(test)

raw_cont = np.dot(layer_norm_output, model.layers[4].get_weights()[0])
raw_contII = raw_cont + model.layers[4].get_weights()[1]
raw_contIII = 1/(1+np.exp(-1*raw_contII))




#### print out
(ttcon.values - model.layers[2].get_weights()[0]) / np.sqrt(model.layers[2].get_weights()[1])




exceled_std = pd.DataFrame(data={'Mean': model.layers[2].get_weights()[0], 'Var': model.layers[2].get_weights()[1]}).T

exceled_logreg_coeffs = pd.DataFrame(data=np.transpose(model.layers[4].get_weights()[0]), columns=ttcon.columns)
exceled_logreg_bias = pd.DataFrame(data=np.transpose(model.layers[4].get_weights()[1]), columns=['Bias'])

exceled_logreg_coeffs_exp = pd.DataFrame(data=np.exp(np.transpose(model.layers[4].get_weights()[0])), columns=ttcon.columns)
exceled_logreg_bias_exp = pd.DataFrame(data=np.exp(np.transpose(model.layers[4].get_weights()[1])), columns=['Bias'])

#to see the effects on a unit increment of the original feature
exceled_logreg_coeffs_std = model.layers[4].get_weights()[0]/np.reshape(np.sqrt(model.layers[2].get_weights()[1]), newshape=[-1, 1])
exceled_logreg_coeffs_exp_std = pd.DataFrame(data=np.exp(np.transpose(exceled_logreg_coeffs_std)), columns=ttcon.columns)
#

exceled_emb = pd.DataFrame(data=np.transpose(np.squeeze(model.layers[3].get_weights())))
summaries


with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
    exceled_std.to_excel(writer, sheet_name='vis', header=False)
    exceled_logreg_coeffs.to_excel(writer, startrow=exceled_std.values.shape[0]+1, sheet_name='vis')
    exceled_logreg_bias.to_excel(writer, startrow=exceled_std.shape[0]+1, startcol=exceled_logreg_coeffs.shape[1]+1, sheet_name='vis', index=False)
    exceled_emb.to_excel(writer, startrow=exceled_std.shape[0]+1,
                                 startcol=exceled_logreg_coeffs.shape[1] + exceled_logreg_bias.shape[1] + 2, sheet_name='vis')
    summaries.T.to_excel(writer, startrow=exceled_std.shape[0]+exceled_emb.shape[0]+2,
                                 startcol=exceled_logreg_coeffs.shape[1] + exceled_logreg_bias.shape[1] + 3, sheet_name='vis')
    exceled_logreg_coeffs_exp.to_excel(writer, startrow=exceled_std.values.shape[0] + exceled_logreg_coeffs.shape[0] + 2, sheet_name='vis')
    exceled_logreg_bias_exp.to_excel(writer, startrow=exceled_std.shape[0] + exceled_logreg_bias.shape[0] + 2, startcol=exceled_logreg_coeffs.shape[1] + 1,
                                 sheet_name='vis', index=False)
    exceled_logreg_coeffs_exp_std.to_excel(writer,
                                       startrow=exceled_std.values.shape[0] + exceled_logreg_coeffs.shape[0] + exceled_logreg_coeffs_exp.shape[0] + 3,
                                       sheet_name='vis')


#PLOT EMBEDDING RESULTS

vis_emb = pd.DataFrame(layer_disc_output)
letters = ["A","B","C","D","E","F","G","H","I","J","K"]
vis_emb.columns = letters[0: vis_emb.columns.__len__()]


#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(vis_emb['X'], vis_emb['Y'], vis_emb['Z'], c='skyblue', s=60)
# ax.view_init(30, 185)
# plt.show()


pg = pd.concat([vis_emb, ttd.reset_index()], axis=1)
#pg = pg.drop(['index'], axis=1)
['A', 'B', 'C', 'D', 'E', 'F', 'Sex', 'Pclass', 'Embarked']

stats = pg.groupby(['A', 'B', 'C', 'D', 'E', 'F', 'Sex', 'Pclass', 'Embarked']).count().reset_index()
del stats
g = sbn.PairGrid(pg, vars=letters[0: vis_emb.columns.__len__()], hue="Sex")
g.map_upper(plt.scatter)
g.map_lower(sbn.kdeplot)
g.map_diag(sbn.kdeplot, lw=3, legend=False);
