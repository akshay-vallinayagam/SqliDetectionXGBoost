# SqliDetectionXGBoost

1) Command to start Flask:
==========================
set FLASK_APP=main.py & set FLASK_DEBUG=1 & flask run

2) cURL command to test the response time of the server:
========================================================
curl -X POST http://localhost:5000/pythonlogin/ -d "username=test&password=test" -w '\nhttp_response_cod e:%{http_code}\ntime_total:%{time_total}\n' -o NUL


3)SQLmap Command to perform automated testing:
==============================================
sqlmap -u "http://localhost:5000/pythonlogin/" --data "username=akshay&password=Project1" --dbms=mysql --batch --ignore-code=403,401 --level=3 --risk=3 --tamper=space2comment --random-agent
