'''
This part does not work yet
the idea is from https://spark.apache.org/docs/latest/ml-classification-regression.html#regression
'''

from pyspark.ml.regression import LinearRegression

# Load training data
filename = 'test.libsvm'
data = spark.read.format("libsvm").load(filename)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# valuesAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
# MSE = valuesAndPreds \
#     .map(lambda (v, p): (v - p)**2) \
#     .reduce(lambda x, y: x + y) / valuesAndPreds.count()
# print("Mean Squared Error = " + str(MSE))

# Print the coefficients and intercept for linear regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))