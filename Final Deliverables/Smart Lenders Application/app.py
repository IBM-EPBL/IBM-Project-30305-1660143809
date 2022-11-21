from flask import Flask, request, render_template

import requests
import pickle
import warnings
from os import environ
warnings.filterwarnings('ignore')

API_KEY = "YOUR_API_KEY_HERE"


token_response = requests.post(
    'https://iam.cloud.ibm.com/identity/token', 
    data = {
        "apikey": API_KEY, 
        "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
    }
)


mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('./model/Smart_Lenders_Support Vector Classifier.pkl','rb'))

scalerfile = 'utils/scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

educationfile = 'utils/Education_encoder.pkl'
education_encoder = pickle.load(open(educationfile, 'rb'))

genderfile = 'utils/Gender_encoder.pkl'
gender_encoder = pickle.load(open(genderfile, 'rb'))

loanfile = 'utils/Loan Status_encoder.pkl'
loan_encoder = pickle.load(open(loanfile, 'rb'))

marriedfile = 'utils/Married_encoder.pkl'
married_encoder = pickle.load(open(marriedfile, 'rb'))

propertyfile = 'utils/Property_Area_encoder.pkl'
property_encoder = pickle.load(open(propertyfile, 'rb'))

employmentfile = 'utils/Self_Employed_encoder.pkl'
employment_encoder = pickle.load(open(employmentfile, 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():

    name = request.form["name"]
    age = int(request.form["age"])
    gender = request.form["gender"]
    married = request.form["married"]
    dependents = int(request.form["dependents"])
    education = request.form["education"]
    employement = request.form["SEmp"]
    aincome = int(request.form["aincome"])
    caincome = int(request.form["caincome"])
    loan = int(request.form["loan"])
    duration = int(request.form["duration"])
    chistory = request.form["chistory"]
    region = request.form["propregion"]
    email = request.form["email"]
    phno_ = request.form["phno"]

    # Transform input data!

    gender = gender_encoder.transform([gender])[0]
    married = married_encoder.transform([married])[0]
    education = education_encoder.transform([education])[0]
    employement = employment_encoder.transform([employement])[0]
    region = property_encoder.transform([region])[0]

    features = [gender, married, dependents, education, employement, aincome, caincome, loan, duration, chistory, region]
    features = scaler.transform([features])
    
    payload_scoring = {"input_data": [{"fields": ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'], "values": features.tolist()}]}

    response_scoring = requests.post('https://eu-de.ml.cloud.ibm.com/ml/v4/deployments/d95fbefa-4503-49cf-b870-1348309c3bdc/predictions?version=2022-11-16', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print(response_scoring.json())

    # print(val)

    response_scoring = response_scoring.json()
    val = response_scoring['predictions'][0]['values'][0][0]
    pred = val

    if pred == 0:
        text = 'Sorry '+ name +', you are not eligible for a loan! Please contact the team for further details.'
    else:
        text = 'Hi '+ name +', you are eligible for a loan! Please contact the team for further information.' 

    # return 'hi'
    return render_template(
        'response.html',
        prediction_text = text, 
        name_text = request.form["name"],
        email_text = request.form["email"],
        phno = request.form["phno"],
        age_text = request.form["age"],
        gender_text = request.form["gender"],
        married_text = request.form["married"],
        dep_text = request.form["dependents"],
        grad_text = request.form["education"],
        chistory_text = request.form["chistory"],
        emp_text = request.form["SEmp"],
        aincome_text = request.form["aincome"],
        caincome_text = request.form["caincome"],
        loan_text = request.form["loan"],
        dur_text = request.form["duration"],
        reg_text = request.form["propregion"]
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=environ.get("PORT", 5000), threaded=True)