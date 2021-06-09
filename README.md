# Customer Feedback Sentiment Analysis 

A `LSTM` Recurrent Neural Network based Flask Web App that classifies the sentiment of Customer Textual Review as Positive or Negative. 

# Getting Started

To get the app working locally:
1. Clone or download the repository locally.
2. Within the Customer-Feedback-Sentiment-Analysis-using-LSTM directory, create a virtual Python environment with the Terminal command `python -m venv flaskapp` where `flaskapp` is the name of your environment. You can choose any name.
3. Activate the virtual environment with the command        `
    ```bash                 
    flaskapp\scripts\activate.bat
    ```
4. Then run the command `pip install -r requirements.txt`
5. Next, set the FLASK_APP variable to app.py and FLASK_ENV to development by running the following command (for windows) 
   ```bash
    set FLASK_APP=app.py
    ```
6. Also, set the FLASK_ENV to `development` by running the following command (for windows)
    ```bash
    set FLASK_ENV=development
    ```
7. And finally, run the command `python -m flask run` to start the app
8. The terminal will output the local web address and port where the app is running. As an example, this might be `http://127.0.0.1:5000/`. Now, open a web browser and go to that web address.

# Prerequisites

You will need [Python3 installed](https://www.python.org/downloads/) on your local machine.

# Built With

* [Python](https://www.python.org/) - Programming language
* [Tensorflow](https://www.tensorflow.org/) - RNN Model
* [Flask](http://flask.pocoo.org/) - Web Development Framework
* [Pandas](https://pandas.pydata.org/) - Data Manipulation and Analysis

# Interface Sample

![image](https://user-images.githubusercontent.com/45168689/121368290-65f62c80-c954-11eb-97ff-6c84a9eba73c.png)

Feel free to give a star to this project if you like and support! Adios

 

