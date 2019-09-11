from app import app
from flask import render_template, request, jsonify
import base64
from model.model import Model

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

import pandas as pd
from io import BytesIO
import time
from app import trainCSV

@app.route('/get_train', methods=['POST'])
def get_train():
    global trainCSV
    params = request.form.to_dict()
    trainCSV += params['text']
    return jsonify({'status' : 'OK'})

@app.route('/get_params', methods=['POST'])
def get_params():
    global trainCSV
    params = request.form.to_dict()
    priceSMS = float(params['priceSMS'])
    priceEmail = float(params['priceEmail'])
    if priceEmail == 0 or priceSMS == 0:
        return jsonify({'status' : 'Not OK'})
    sum_to = float(params['sum'])
    withSMS = params['withSMS'] == 'true'
    withEmail = params['withEmail'] == 'true'
    testCSV = BytesIO(base64.b64decode(params['fileTest']))
    model = Model()
    if trainCSV == '':
        model.load_prev()
    else:
        trainCSV = BytesIO(base64.b64decode(trainCSV))
        model.fit(trainCSV)
    model.apply(testCSV)
    emails, smss, percents, responses, x = model.optimize(priceSMS, priceEmail, sum_to, withSMS, withEmail)
    return jsonify({'status' : 'OK', 'emails' : emails, 'smss' : smss, 'percents':percents, 'responses':responses, 'x':x})

@app.route('/check_validity', methods=['POST'])
def check_validity():
    params = request.form.to_dict()
    variables = ['ID', 'Age', 'Ind_Household', 
                 'Age_group', 'District', 'Region',
                 'Lifetime', 'Income', 'Segment',
                 'Ind_deposit', 'Ind_salary', 'trans_6_month',
                 'trans_9_month', 'trans_12_month', 'amont_trans',
                 'amont_day_from', 'trans_3_month', 'Gender']
    if len(variables) != len(params):
        return jsonify({'status' : 'Not OK'})
    for i in range(len(variables)):
        if (params[str(i)] != variables[i]):
            return jsonify({'status' : 'Not OK'})
    return jsonify({'status' : 'OK'})
