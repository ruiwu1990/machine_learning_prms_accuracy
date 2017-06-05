'''
The ideas are from https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
'''

import pyspark.sql
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import sys

from pyspark.sql import SQLContext
from pyspark import SparkContext
import math
from scipy import stats
import numpy as np
from itertools import izip

def collect_features(input):
	'''
	this function is used to collect delta errors from array
	'''
	output = []
	for i in input:
		output.append(i['label'])
	return output

def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

def collect_id(input_row_list):
	'''
	this function gets the id from the input
	pyspark Row list and return id list and pyspark
	Row without id
	'''
	id_list = []
	row_list = []
	for i in input_row_list:
		# last item of the array is id
		id_list.append(i['features'].toArray()[-1])
		# features without id
		tmp_features = i['features'].toArray()[:-1]
		# create sparse dict
		tmp_dict = dict(izip(range(len(tmp_features)),tmp_features))
		new_row_feature = pyspark.ml.linalg.SparseVector(len(tmp_features),tmp_dict)
		row_list.append(pyspark.sql.types.Row(label=i['label'], features=new_row_feature))
	return id_list, row_list

def find_min_label(input_row_list):
	'''
	this function finds the min label value
	'''
	tmp_min = 100000
	# obtain min label value
	for i in input_row_list:
		if i['label'] < tmp_min:
			tmp_min = i['label'] 
	return tmp_min

def log_sinh_transform(input_row_list,tmp_min,epsilon = 0.01):
	'''
	this function transform original y
	it returns transformed value and a (tmp_min)
	'''
	row_list = []
	for i in input_row_list:
		# transform here
		# print '+++++++++++++++++++'+str(tmp_min)
		# print '-------------------'+str()
		# print "/////////////////////////////////////"+str(math.sinh(i['label'])-tmp_min+epsilon)
		# when x >o, sinh(x) >0
		row_list.append(pyspark.sql.types.Row(label=math.log10(math.sinh(i['label']-tmp_min+epsilon)), features=i['features']))
	return row_list

def reverse_log_sinh_transform(input_list,tmp_min,epsilon = 0.01):
	'''
	this function transform y back
	'''
	result_list = []
	for i in input_list:
		# transform here
		# http://mathworld.wolfram.com/InverseHyperbolicSine.html
		sinh_part = 10**i
		tmp_origin = math.log(sinh_part+math.sqrt(1+sinh_part*sinh_part))+tmp_min-epsilon
		result_list.append(tmp_origin)
	return result_list

sc = SparkContext()
sqlContext = SQLContext(sc)

# Load the data stored in LIBSVM format as a DataFrame.
# filename = 'static/data/test.libsvm'
filename = sys.argv[1]
data = sqlContext.read.format("libsvm").load(filename)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([float(sys.argv[3]), 1-float(sys.argv[3])])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# changes !!!!!!!!!!!!!!!!
tmp_min = find_min_label(data.collect())
# convert dataframe into list
test = testData.collect()
test = log_sinh_transform(test,tmp_min)
# test_id, test = collect_id(test)
train = trainingData.collect()
train = log_sinh_transform(train,tmp_min)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~not done
# changes !!!!!!!!!!!!!!!! end

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
# predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# write the results into the file
fp = open(sys.argv[2],'a')

fp.write(str(rmse)+'\n')
# fp.write(sys.argv[3]+": aaa "+str(rmse)+'\n')

fp.close()

# treeModel = model.stages[1]
# # summary only
# print(treeModel)