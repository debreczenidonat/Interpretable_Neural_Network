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



tuple = initial_transformation_for_model(test)
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

### LOADING MODEL
new_model = load_model('intNNmodelsave.h5', custom_objects={'custom_activity_regIV': custom_activity_regIV})

test_merge = [ttcon.values.astype('float32'), ttcoded.values.astype('float32')]
new_model.predict(test)
#
# @app.route('/aaa/<int:id>')
# def print_id(id):
#     return('Hello %s' % id)
#
# @app.route('/bbb/<guest>')
# def hello_world(guest):
#     return('Hello %s' % guest)
#
# @app.route('/redirector/<name>')
# def redir(name):
#     if name=='Aladar':
#         return(redirect(url_for('print_id', id=2)))
#     else:
#         return(redirect(url_for('hello_world', guest=name)))
#
# @app.route('/success/<name>')
# def success(name):
#    return 'welcome %s' % name
#
# @app.route('/login',methods = ['POST', 'GET'])
# def login():
#    if request.method == 'POST':
#       user = request.form['nm']
#       return redirect(url_for('success',name = user))
#    else:
#       user = request.args.get('nm')
#       return redirect(url_for('success',name = user))


# @app.route('/')
# def student():
#    return render_template('student.html')
#
# @app.route('/result',methods = ['POST', 'GET'])
# def result():
#    if request.method == 'POST':
#       result = request.form
#       return render_template("result.html",result = result)
#
# if __name__ == '__main__':
#    app.run()
