# Udacity Project: Disaster Response Pipelines

<a id='top'></a>

## Table of Contents
1. [Introduction](#introduction)

2. [File Descriptions](#file)

3. [Quick Start](#start)

4. [Note](#note)

5. [Acknowledgements](#acknowledgement)

<a id='introduction'></a>

## Introduction

In this project, we analyze disaster data from **Figure Eight** to build a model for an API that classifies disaster messages. The datasets contain real messages that were sent during disaster events. We build a machine learning natural language processing pipeline to categorize these events. In addition, the project includes a web app where an emergency worker can input a new message and get classification results in several categories.

<a id='file'></a>

## File Description

The file structure of the project:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model (the pickle file on local machine is too large to be uploaded, 
                                  so we do not have pkl file in this git repository)

- README.md
```

<a id='start'></a>

## Quick Start

To run the web app, follow the steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. On your local machine, open a browser and go to http://localhost:3001/

<a id='note'></a>

## Note 

Note that we do not deploy the web app to a cloud service provider. Instead, we will provide some screenshots of the web app.

![main screen](/screenshot/1.png)
![data visualization](/screenshot/2.png)
![message classification](/screenshot/3.png)
 
<a id='acknowledgement'></a>

## Acknowledgement

The data of the project were offered by **Figure Eight** and the project was based on Udacity degree.

[Back to top](#top)
