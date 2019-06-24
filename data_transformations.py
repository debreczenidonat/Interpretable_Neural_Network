import pandas as pd
import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer

def step_i_creating_discrete_features(tt):
    tt['IsChild'] = tt['Age'] <= 10
    tt['IsChild'] = tt['IsChild'].apply(lambda x: str(x))
    tt['codedcats'] = tt['Pclass'].apply(lambda x: str(x)) + tt['Sex'] + tt['Embarked'] + tt['IsChild']
    return(tt)


def step_ii_code_discrete(tt, tok):
    tt['codedcats'] = tok.texts_to_sequences(tt['codedcats'])
    tt['codedcats'] = tt['codedcats'].apply(lambda x: np.squeeze(x).astype(float))
    tt['codedcats'] = pd.to_numeric(tt['codedcats'])
    return(tt)


def step_iii_selecting_var_types(i, cont_vars=[], discrete_vars=[], coded_var=[], target_var=[]):
    ttcon = i[cont_vars]
    ttd = i[discrete_vars]
    ttcoded = i[coded_var]
    tttarget = i[target_var]
    return ttcon, ttd, ttcoded, tttarget


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


# def custom_activity_regIV(vects):
#     a = K.abs(K.mean(K.identity(vects)) / K.max(K.identity(vects)))
#     b = K.abs(K.sum(K.identity(vects)) / K.max(K.identity(vects)))
#     c = K.abs(K.max(K.identity(vects)) - 1)
#     d = K.abs(K.sum(K.identity(vects)) - 1)
#     return 0.2*(K.log(b) + K.log(1+c))


class custom_activity_regIV(Regularizer):
    def __init__(self, ll=0.2):
        self.ll = ll

    def __call__(self, x):
        b = K.abs(K.sum(K.identity(x)) / K.max(K.identity(x)))
        c = K.abs(K.max(K.identity(x)) - 1)
        return self.ll * (K.log(b) + K.log(1 + c))

    def get_config(self):
        return {'ll': float(self.ll)}
