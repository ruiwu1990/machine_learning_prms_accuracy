This project is a machine learning application using Apache Spark ML.Model accuracy is one of the most important properties of a model. Even two or three percentages of accuracy could mean millions of dollars or the difference between life and death. Machine learning is commonly used to build accurate models, and many tools, such as Weka, exist for such purposes. However, when the dataset is huge or when the modeling task varies greatly from domain to domain, traditional tools are limited and different machine learning models are needed to be chosen for different problems. Big data and associated computational challenges can present obstacles to building accurate models. There are popular libraries that facilitate big data machine learning problems, such as the Apache Spark Machine Learning library. However, it requires the user to be able to program and use the offered API to obtain accurate results. In this paper, we propose a general way to improve the accuracy of a model without exploring model theory. In theory, the method should work for any arbitrary model with sufficient output. The basic idea is to predict the differences (delta) between the observed values and the model predicted values. We also created a user-friendly web-based system that leverages Apache Spark to process large data and build such a model. The Precipitation Runoff Modeling System, a physical hydrology model, was used to test our system. We were able to improve the original model’s error in terms of multiple statistical metrics using four out of five different machine learning models that we applied. The framework operates on model output alone, therefore it offers a potentially robust solution to improving the accuracy of any arbitrary model given sufficient output.


Here are the ML techniques used from Spark Tool:
```
http://spark.apache.org/docs/latest/mllib-classification-regression.html
```
#Quick Start
Here is the command to install the requirements
```
sudo pip install requirements.txt
```
Here is the command to set up and run the program
```
python views.py -h 134.197.20.79 -p 5000 --threaded
```
134.197.20.79 should be replaced with your machine ip address. The command is to set up a server with your machine


This tool is completely dockerized. Now you can deploy this tool on any machine which is configured with a docker engine.
To deploy the tool, use the below docker command.
```
docker run --name <container_name> -p 5022:5000 josepainumkal/machine_learning_prms_accuracy:jose_thesis python views.py
```
To deploy the tool, with a volume ( for development/debugging):
```
docker run -v /home/docker/machine_learning_prms_accuracy:/regtool --name <container_name> -p 5022:5000 josepainumkal/machine_learning_prms_accuracy:jose_thesis python views.py
```
In the above command, "/home/docker/machine_learning_prms_accuracy" is the location of the tool's repository in container's host machine. Replace it with your repository location. 

Thank you !!! 














