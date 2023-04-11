# Tutorial-on-creating-a-dockerFile-
Tutorial on creating a dockerFile for a python training file of a sample ML code 

# Docker Basic (Gist) 


 * Docker is a platform that allows developers to automate the deployment by unning of applications inside lightweight, portable and self-sufficient containers. 

* Containers are standalone, executable software packages that include everything needed to run an application, including the code, runtime, system tools, and libraries.

To read More on containers: [link](https://docker-curriculum.com/#what-are-containers-)

# Practical use case 

Docker allows you to create lightweight and portable containers that encapsulate your application and all its dependencies, including libraries, runtime environments, and other dependencies. This means that you can build and package your application along with its dependencies into a Docker container, and then run the container on any host system that has Docker installed, without having to install those dependencies directly on your local computer.

### Let's Take an example  i have a file training.py i need to make a docker file for lets build it 

our training code may look like this 



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

social_N_data = pd.read_csv('Social_Network_Ads.csv')
pd.concat([social_N_data.head(), social_N_data.tail()])

#CHECK FOR NULL VALUES
social_N_data.isnull().any()

# CLEAN THE DATA
social_N_data.drop('User ID', axis=1, inplace=True)

# CHANGE CATEGORICAL VARIABLE TO DUMMIES
social_N_data.info()
gender = pd.get_dummies(social_N_data['Gender'], drop_first=True)
social_N_data.drop('Gender',axis=1,inplace=True)
social_N_data = pd.concat([social_N_data,gender], axis=1)

# SPLIT DATA TO INDEPENDENT AND DEPENDENT VARIABLES
X = social_N_data.iloc[:,[0,1,3]] # Age, EstimatedSalary and Male
y = social_N_data.iloc[:, 2] # Purchased

# FEATURE SCALING
sc = StandardScaler()
X = sc.fit_transform(X)

# SPLIT DATA TO TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# FIT/TRAIN MODEL
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# PREDICTIONS
y_pred = classifier.predict(X_test)
result = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
print(result)

# EVALUATE MODEL
# predic_proba()
# print(classifier.predict_proba(X) # uncheck if needed
#confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix \n', cf_matrix)

sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print('Accuracy of model')
print(accuracy_score(y_test,y_pred) * 100, '%')
#0.8083333333333333

# classification report
target_names = ['will NOT PURCHASE', 'will PURCHASE']
print('Classification report: \n', classification_report(y_test, y_pred,target_names=target_names))
```

#### Now Lets Make a Dokerfile 

* Step 1:
Create a file with noextention justname it DOCKERFILE.

* Step 2:
In this Dockerfile, we're starting with the official Python 3.8 base image from Docker Hub. 

* Step 3:
Set the working directory to /app inside the container.

* Step 4:
Copy the training.py file into the container.

* Step 5:
And then install any dependencies using pip (e.g., numpy and scikit-learn) or wirte them on a requrement.txt file and then copy them as requrement.txt file.

* Step 6:
We also define an environment variable MODEL_FILE which can be used to specify the model file used by the application.

* Step 7:
And expose port 8000, assuming that the application might be listening on that port. 

* Step 8:
Finally, we set the command to run the application using CMD with python command and passing training.py as the argument.




```python
# Use the official Python base image with a specified version
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY training.py .

# Install any dependencies needed for the application
RUN pip install numpy pandas seaborn matplotlib scikit-learn  

# Define any environment variables if necessary
ENV MODEL_FILE=model.pkl

# Expose any necessary ports for the application
EXPOSE 8000

# Run the application
CMD ["python", "training.py"]

```


```python
# This is the code with requrements.txt with it 

# Use a base image with Python 3.7 installed
FROM python:3.7

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the machine learning code to the container
COPY your_ml_code.py .

# Run the machine learning code
CMD ["python", "your_ml_code.py"]

```

##### Finally 
 
 You can build the Docker image using the following command from the same directory where your Dockerfile is located:

###### Command 

docker build -t my-ml-app



```python
 Read more on : [link](https://www.educative.io/answers/how-do-you-write-a-dockerfile)
```
