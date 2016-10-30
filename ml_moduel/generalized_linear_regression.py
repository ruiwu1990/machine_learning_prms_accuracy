from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark import SparkContext
import json
import sys
# fp =open(sys.argv[2],'w')
# fp.write('asjkhkjh')
# fp.close()
sc = SparkContext()
sqlContext = SQLContext(sc)



# Load training data
filename = sys.argv[1]
dataset = sqlContext.read.format("libsvm").load(filename)

glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

# Fit the model
model = glr.fit(dataset)

# Print the coefficients and intercept for generalized linear regression model
print("\n\n\n\nCoefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
summary = model.summary
# print("\n\n\n\nCoefficient Standard Errors: " + str(summary.coefficientStandardErrors))
# print("T Values: " + str(summary.tValues))
# print("P Values: " + str(summary.pValues))
# print("Dispersion: " + str(summary.dispersion))
# print("Null Deviance: " + str(summary.nullDeviance))
# print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
# print("Deviance: " + str(summary.deviance))
# print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
# print("AIC: " + str(summary.aic))
# print("Deviance Residuals: ")
summary.residuals().show()

# model.transform(dataset).show(truncate=False)

predictions = model.transform(dataset)
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

fp = open(sys.argv[2],'w')
fp.write("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
fp.write('\n')

predictions = model.transform(dataset)
pd_df = predictions.toPandas()
fp.write(','.join([str(i) for i in pd_df['prediction'].tolist()]))

fp.close()

# print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# print ("/////////////////////////////////////////////////////////////////////PREDICTION FOR 0.003251")
# print predictions.predict(0.003251)
# numeric_filtered_1 = predictions.where(predictions['label'] == '0.003251')
# #numeric_filtered_1.foreach(print)
# val = numeric_filtered_1.select('prediction')
# jsonStr = val.toJSON().first()
# resp_dict = json.loads(jsonStr)
# print (jsonStr)
# print(resp_dict['prediction'])
