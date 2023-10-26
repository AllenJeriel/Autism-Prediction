import mysql
from flask import Flask,render_template,url_for,request
from mysql.connector import cursor
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC
import seaborn as sns
import pickle
mydb = mysql.connector.connect(host='localhost',user='root',password="Tamilan@12",port='3306',database='autism')
import matplotlib.pyplot as plt
app=Flask(__name__)
def preprocessing(file):
    file['Class'].replace('No',0,inplace=True)
    file['Class'].replace('Yes',1,inplace=True)
    file['Sex'].replace('m',0,inplace=True)
    file['Sex'].replace('f',1,inplace=True)
    file['Jaundice'].replace('no',0,inplace=True)
    file['Jaundice'].replace('yes',1,inplace=True)
    file['Family_mem_with_ASD'].replace('no',0,inplace=True)
    file['Family_mem_with_ASD'].replace('yes',1,inplace=True)
    file['Who_completed_the_test'].replace('Health Care Professional','Health care professional',inplace=True)
    file['Who_completed_the_test'].replace('family member',0,inplace=True)
    file['Who_completed_the_test'].replace('Health care professional',1,inplace=True)
    file['Who_completed_the_test'].replace('Self',2,inplace=True)
    file['Who_completed_the_test'].replace('Others',3,inplace=True)
    return file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registration',methods=['POST','GET'])
def registration():
    if request.method=="POST":
        print('a')
        un=request.form['name']
        print(un)
        em=request.form['email']
        pw=request.form['password']
        print(pw)
        cpw=request.form['cpassword']
        if pw==cpw:
            sql = "SELECT * FROM hmg"
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails=cur.fetchall()
            mydb.commit()
            all_emails=[i[2] for i in all_emails]
            if em in all_emails:
                return render_template('registration.html',msg='a')
            else:
                sql="INSERT INTO hmg(name,email,password) values(%s,%s,%s)"
                values=(un,em,pw)
                cursor=mydb.cursor()
                cur.execute(sql,values)
                mydb.commit()
                cur.close()
                return render_template('registration.html',msg='success')
        else:
            return render_template('registration.html',msg='repeat')
    return render_template('registration.html')

@app.route('/login',methods=["POST","GET"])
def login():
    if request.method=="POST":
        em=request.form['email']
        print(em)
        pw=request.form['password']
        print(pw)
        cursor=mydb.cursor()
        sql = "SELECT * FROM hmg WHERE email=%s and password=%s"
        val=(em,pw)
        cursor.execute(sql,val)
        results=cursor.fetchall()
        mydb.commit()
        print(results)
        print(len(results))
        if len(results) >= 1:
            return render_template('home.html',msg='login usuccesful')
        else:
            return render_template('login.html',msg='Invalid Credentias')


    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    global df
    if request.method=="POST":
        file=request.files['file']
        print(type(file.filename))
        print('hi')
        df=pd.read_csv(file)
        df = df.drop('Case_No', axis=1)
        print(df.head(2))
        return render_template('upload.html', msg='Dataset Uploaded Successfully')
    return render_template('upload.html')

@app.route('/view_data')
def view_data():
    print(df)
    print(df.head(2))
    print(df.columns)
    return render_template('viewdata.html',columns=df.columns.values,rows=df.values.tolist())
@app.route('/split',methods=["POST","GET"])
def split():
    global X,y,X_train,X_test,y_train,y_test
    if request.method=="POST":
        size=int(request.form['split'])
        size=size/100
        print(size)
        dataset=preprocessing(df)
        print(df)
        print(df.columns)
        X = dataset.drop('Class', axis=1)
        y = dataset['Class']
##        dataset = dataset.loc[0:364]
##        X=dataset.drop(['HGB','S. No.'],axis=1)
##        y=dataset['HGB']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=size,random_state=52)
        print(y_test)

        return render_template('split.html',msg='Data Preprocessed and It Splits Succesfully')
    return render_template('split.html')

# Define the list to store model results
comparison_results = []

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        model_choice = int(request.form['algo'])
        
        if model_choice == 0:
            return render_template('model.html', msg='Please Choose any Algorithm')

        if model_choice == 1:
            model = RandomForestClassifier(n_estimators=600)
            model_name = 'Random Forest'
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('RF_Confusion Matrix')
            rfcon_filename = 'static/Random_forest_confusion_matrix.png'
            plt.savefig(rfcon_filename)  # Save the plot as an image file
        else:
            model = SVC()
            model_name = 'Support Vector Machine'
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('SVM_Confusion Matrix')
            svmcon_filename = 'static/SVM_confusion_matrix.png'
            plt.savefig(svmcon_filename)  # Save the plot as an image file
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred) * 100

        # Create a dictionary to store the model name and accuracy
        results = {
            'Model': model_name,
            'Accuracy': accuracy
        }

        # Append the results to the list
        comparison_results.append(results)
                # Create a bar chart to compare accuracy scores
        models = [result['Model'] for result in comparison_results]
        accuracies = [result['Accuracy'] for result in comparison_results]

        # Save the plot to a file or display it in the Flask app
        plt.savefig('comparison_chart.png')  # You can save it as an image file
        # or use plt.show() to display it in the app

        return render_template('model.html', msg=f'The Accuracy Score for {model_name} is {accuracy:.2f}%')


    return render_template('model.html')


@app.route('/analysis', methods=['GET'])
def analysis():
    if comparison_results:
        # Create a bar chart to compare accuracy scores
        models = [result['Model'] for result in comparison_results]
        accuracies = [result['Accuracy'] for result in comparison_results]

        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Comparison')
        plt.ylim(0, 100)

        # Save the plot to a file or display it in the Flask app
        chart_filename = 'static/comparison_chart.png'
        plt.savefig(chart_filename)  # Save it as an image file

        return render_template('analysis.html', chart_filename=chart_filename)

    return render_template('analysis.html', msg='No comparison data available.')

@app.route('/prediction',methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        q1=int(request.form['q1'])
        q2=int(request.form['q2'])
        q3=int(request.form['q3'])
        q4=int(request.form['q4'])
        q5=int(request.form['q5'])
        q6=int(request.form['q6'])
        q7=int(request.form['q7'])
        q8=int(request.form['q8'])
        q9=int(request.form['q9'])
        q10=int(request.form['q10'])
        q11=int(request.form['q11'])
        q12=int(request.form['q12'])
        q13=int(request.form['q13'])
        q14=int(request.form['q14'])
        q15=int(request.form['q15'])
        l=[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15]
        print(l)
        model=RandomForestClassifier(n_estimators=600)
        model.fit(X_train,y_train)
        ot=model.predict([l])
        print(ot)
        if ot==1:
            a='The child may have Autism Spectrum Disorder'
        else:
            a="The child does't have Autism Spectrum Disorder"
        return render_template('prediction.html',msg=a,re=ot)
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True)
