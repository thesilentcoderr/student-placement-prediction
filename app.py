import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model_rf = pickle.load(open('randomforestentropymodel.pkl','rb')) 
model_gb = pickle.load(open('gaussianmodel.pkl','rb'))
model_lr = pickle.load(open('model_lr.pkl','rb'))
model_svcl = pickle.load(open('model_svcl.pkl','rb'))
model_dt = pickle.load(open('model_dt.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    _10thclass = float(request.args.get('ten'))
    _12thclass = float(request.args.get('tw'))
    btech = float(request.args.get('bt'))
    _7th = float(request.args.get('seven'))
    _6th = float(request.args.get('six'))
    _5th = float(request.args.get('five'))
    final = float(request.args.get('final'))
    medium = float(request.args.get('medium'))
    package = float(request.args.get('pack'))
    
    

    
    
    if request.form.get('rf') == 'rf':
        prediction = model_rf.predict([[_10thclass,_12thclass,btech,_7th,_6th,_5th,final,medium,package]])
    elif request.form.get('lr') == 'lr':
        prediction = model_lr.predict([[_10thclass,_12thclass,btech,_7th,_6th,_5th,final,medium,package]])
    elif request.form.get('svcl') == 'svcl':
        prediction = model_svcl.predict([[_10thclass,_12thclass,btech,_7th,_6th,_5th,final,medium,package]])
    elif request.form.get('dt') == 'dt':
        prediction = model_dt.predict([[_10thclass,_12thclass,btech,_7th,_6th,_5th,final,medium,package]])
    else:
      prediction = model_gb.predict([[_10thclass,_12thclass,btech,_7th,_6th,_5th,final,medium,package]])
    if prediction==[1]:
      prediction_text="Model  has predicted that the student is Placed :{}".format(prediction)
    else:
      prediction_text="Model  has predicted that the person is Not Placed :{}".format(prediction)
        
    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run()