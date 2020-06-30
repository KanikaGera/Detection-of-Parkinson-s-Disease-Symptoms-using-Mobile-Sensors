The project has an mobile application which collects the data from the user and uses the data collected to predict the presence of parkinson's symptoms.
Parkinson can have many symptoms, but this app specifically focusses on tremor as a symptom.
Data collected from users is preprocessed and feature engineered . The processed data is placed in Data Folder.
The design used to collect data is

Feature Engineering notebooks and Exploratory Data Analysis Notebooks are in "Machine Learning Code" subfolder in Code. It consist of ML.ipynb which uses the 
feature engineered data , and various ML algorithms to predict presense of Tremor as Parkinson Disease Symptom.
 
Trained Model _ Prediction consist of notbook that collects data calls a lambda function hosted on AWS, that reads new tuples of raw datafrom dynamo db,
perform feature engineering automatically , passes through the best trained model and return the prediction.