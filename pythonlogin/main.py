# Authors: Akshay Vallinayagam (avallina@nyit.edu), Saikrishna Sunkara(vsunkara@nyit.edu) 
# This module contains the code for web application and uses the trained XGBoost classifier model to detect SQLi injection in the web application
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
from model import convert_To_Lower_case, create_features
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

import pandas as pd
import MySQLdb.cursors
import re
import pickle
import logging
from datetime import datetime

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

#Logging configuration
logger = logging.getLogger('myApp')
hdlr = logging.FileHandler('logs\\log_{}.log'.format(datetime.strftime(datetime.now(), '%Y_%m_%d')))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


# Enter your database connection details below (Here the credentials are hardcoded for testing purpose)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Project123$'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

#Load the trained model
model = pickle.load(open('model.pkl','rb'))
train_comment_features = pickle.load(open('train_comment_features.pkl', 'rb'))

# http://localhost:5000/pythonlogin/ - the following will be our login page, which will use both GET and POST requests
@app.route('/pythonlogin/', methods=['GET', 'POST'])
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        Query = "SELECT * FROM accounts WHERE username = {} AND password = {}".format(username, password)
        #print(Query)
        if validate_live_query(Query) == 1:
            logger.error(f'SQL Injection detected for user {username} Query: {Query}')
            return 'SQL INJECTION DETECTED', 403
        #else:
            #print("No SQLI detected")
        #cursor.execute(Query)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            #return 'Logged in successfully!'
            logger.info(f'{username} logged in successfully')
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            logger.info(f'Login failed - Incorrect username/password ')
            return 'Incorrect username/password!', 401
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)
    
# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   uname=session['username']
   logger.info(f'{uname} Logged out successfully')
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
   
   
   
# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            logger.info(f'User registration failed. Reason: {msg}')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            logger.info(f'User registration failed. Reason: {msg}')
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
            logger.info(f'User registration failed. Reason: {msg}')
        elif not username or not password or not email:
            msg = 'Form empty'
            logger.info(f'User registration failed. Reason: {msg}')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            logger.info(f'New user {username} registered successfully')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
            
        
# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
#Feature extraction of the input query and prediction using the imported model
def validate_live_query(Query1):
    TestQuery1 = convert_To_Lower_case(Query1)
    Testdata = [{'Query':TestQuery1}]
    X_test = pd.DataFrame(Testdata)
    X_test = create_features(X_test)
    X_test_query = train_comment_features.transform(X_test['Query'])
    X_test = X_test.drop(['Query'],axis =1 )
    X_test = hstack((X_test,X_test_query)).tocsr()
    return model.predict(X_test)[0]
    

