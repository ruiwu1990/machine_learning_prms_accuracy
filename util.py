import numpy as np
import netCDF4

def get_run_off(input_file):
	'''
	this function gets the run_off (observed data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	observed_data = list(prms_fp.variables['runoff_1'][:])
	prms_fp.close()
	return observed_data

def get_basin_cfs(input_file):
	'''
	this function gets the basin_cfs (predicted data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	predicted_data = list(prms_fp.variables['basin_cfs_1'][:])
	prms_fp.close()
	return predicted_data

def get_time(input_file):
	'''
	this function gets the basin_cfs (predicted data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	time_data = list(prms_fp.variables['time'][:])
	prms_fp.close()
	return time_data

def get_all_input(data_file,param_file,data_len):
	'''
	this function get all the params into a csv file
	as long as the params len equals data_len
	'''
	prms_data_fp = netCDF4.Dataset(data_file,'r')
	prms_param_fp = netCDF4.Dataset(param_file,'r')
	prms_data_fp.close()
	prms_param_fp.close()


def output_csv(input_file):
	'''
	output time, observed data, and predicted data, delta error (observed - predicted)
	into a csv file
	'''
	fp = open('data.csv','w')
	time_list = get_time(input_file)
	predicted_list = get_basin_cfs(input_file)
	observed_list = get_run_off(input_file)
	list_len = len(time_list)
	for count in range(list_len):
		fp.write(','.join([str(time_list[count]),str(predicted_list[count]),str(observed_list[count]),str(observed_list[count]-predicted_list[count])])+'\n')
	fp.close()





input_file = 'stat.nc'
data_file = 'LC.data.nc'
param_file = 'LC.param.nc'
# hard coded here
data_len = 2922
output_csv(input_file)

# the following part convert csv into libsvm
# the function is basically 
# from https://github.com/zygmuntz/phraug/blob/master/csv2libsvm.py
import sys
import csv
from collections import defaultdict

def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )

	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

def convert_csv_into_libsvm(input_file,output_file,label_index=0,skip_headers=True):
	'''
	the function converts csv into libsvm
	'''
	i = open( input_file, 'rb' )
	o = open( output_file, 'wb' )
	reader = csv.reader( i )

	if skip_headers:
		headers = reader.next()

	for line in reader:
		if label_index == -1:
			label = '1'
		else:
			label = line.pop( label_index )

		new_line = construct_line( label, line )
		o.write( new_line )
#


# spark part starts here

# linear regression
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
def parsePoint(line):
	'''
	The function is inspired from 
	http://spark.apache.org/docs/latest/mllib-linear-methods.html
	'''
	values = [float(x) for x in line.strip().split(',')]
	return LabeledPoint(values[0],values[1:])

data = sc.textFile(filename)
parsedData = data.map(parsePoint)

model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)

valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds \
    .map(lambda (v, p): (v - p)**2) \
    .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))


# decision tree regression
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

def parsePoint(line):
	'''
	The function is inspired from 
	http://spark.apache.org/docs/latest/mllib-linear-methods.html
	'''
	values = [float(x) for x in line.strip().split(',')]
	return LabeledPoint(values[0],values[1:])

raw_data = sc.textFile(filename)
data = raw_data.map(parsePoint)

featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

treeModel = model.stages[1]
# summary only
print(treeModel)
# spark part ends here


# this is how to print content in RDD
# from __future__ import print_function
# data.foreach(print)

# # this is how to print column in the dataFrame
# testData.rdd.map(lambda r: r.label).collect()

# def parsePoint(labeledData):
#     return (labeledData.label, labeledData.features[1], labeledData.features[2], 1.0)