from flask import Flask,  redirect, url_for, request, render_template

import pandas as pd
import numpy as np
import pickle
from data_transformations import step_i_creating_discrete_features, step_ii_code_discrete, step_iii_selecting_var_types
from data_transformations import custom_activity_regIV
from keras.regularizers import Regularizer

from keras.models import model_from_yaml, load_model

app = Flask(__name__)

test = (1, 'male', 30, 0, 0, 10, 'S')


def initial_transformation_for_model(input_list):
    input_list = (-1,) + input_list  # Creating dummy col
    input_list = [input_list]
    columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp',  'Parch', 'Fare', 'Embarked']
    the_types = {'Survived': 'int64', 'Pclass': 'int64','Sex': 'object','Age': 'float64','SibSp': 'int64','Parch': 'int64','Fare': 'float64','Embarked': 'object'}
    df = pd.DataFrame.from_records(input_list, columns=columns)
    df.astype(the_types)
    return(df)


def transformations_for_the_model(tuple):
    tuple = initial_transformation_for_model(tuple)
    tuple = step_i_creating_discrete_features(tuple)

    with open('tokenizer.pickle', 'rb') as t:
        tok = pickle.load(t)

    tuple = step_ii_code_discrete(tuple, tok)

    ###CATEGORIZING VARS
    CONT_VARS = ['Age', 'SibSp', 'Parch', 'Fare']
    DISC_VARS = ['Sex', 'Pclass', 'Embarked', 'IsChild']
    CODED_VAR = ['codedcats']
    TARGET_VAR = ['Survived']

    ttcon, ttd, ttcoded, tttarget = step_iii_selecting_var_types(tuple, CONT_VARS, DISC_VARS, CODED_VAR, TARGET_VAR)

    del ttd, tttarget
    merged = [ttcon.values.astype('float32'), ttcoded.values.astype('float32')]
    return merged

test = transformations_for_the_model(test)

### LOADING MODEL
new_model = load_model('intNNmodelsave.h5', custom_objects={'custom_activity_regIV': custom_activity_regIV})

new_model.predict(test)


@app.route('/')
def passenger_creator():
   return render_template('passenger_creator.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      raw_input = request.form
      print(raw_input)
      raw_input = [i for i in raw_input.values()]
      input = (int(raw_input[0]), raw_input[1], int(raw_input[2]), int(raw_input[3]), int(raw_input[4]), int(raw_input[5]), raw_input[6])
      input = transformations_for_the_model(input)
      print(input)
      prediction = new_model.predict(input)
      return render_template("result.html", result = raw_input, prediction = prediction)

if __name__ == '__main__':
   app.run()

# < table
# border = 1 >
# { %
# for key, value in result.items() %}
# < tr >
# < th > {{key}} < / th >
# < td > {{value}} < / td >
# < / tr >
# { % endfor %}
# < / table >