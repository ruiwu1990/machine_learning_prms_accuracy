'''
The ideas are from https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
'''

from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
import sys

from pyspark.sql import SQLContext
from pyspark import SparkContext
import math

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
(trainingData, testData) = data.randomSplit([0.625, 0.375])

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

'''
use union to add first col in testData
need to convert row into df
need to remove first row in testData, using original.subtract(firstRowDF)
'''
# convert dataframe into list
test = testData.collect()
train = trainingData.collect()
# sort by dtime
test = sorted(test, key = lambda x: x['features'][5])
new_train_data = sorted(train, key = lambda x: x['features'][5])
predictions_list = []
ground_truth_list = []
test_len = len(test)
for count in range(test_len):
	current_row = test[count]
	new_data = sc.parallelize([current_row]).toDF()
	# get current prediction
	predictions = model.transform(new_data)	
	# collect predictions into result lists
	predictions_list.append(predictions.toPandas()['prediction'].tolist()[0])
	ground_truth_list.append(predictions.toPandas()['label'].tolist()[0])
	new_train_data = [current_row] + new_train_data
	# remove the oldest data
	new_train_data = sorted(new_train_data, key = lambda x: x['features'][5])[1:]
	new_train_data_df = sc.parallelize([current_row] + new_train_data).toDF()
	# train model again with the updated data
	model = pipeline.fit(new_train_data_df)
	# print out hint info
	print "current processing: "+str(count+1)+"; "+str(test_len-count-1)+" rows left."

# print get_root_mean_squared_error(predictions_list,ground_truth_list)


def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)



# print sys.argv[1]
# print sys.argv[2]
# print sys.argv[3]
# print sys.argv[4]
fp = open(sys.argv[2],'a')
fp.write(str(get_root_mean_squared_error(predictions_list,ground_truth_list))+'\n')
fp.close()