'''
the ideas are from https://spark.apache.org/docs/latest/mllib-linear-methods.html
'''

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

def parsePoint(line):
	'''
	The function is inspired from 
	http://spark.apache.org/docs/latest/mllib-linear-methods.html
	'''
	values = [float(x) for x in line.strip().split(',')]
	return LabeledPoint(values[0],values[1:])

filename = 'static/data/test.csv'

data = sc.textFile(filename)
parsedData = data.map(parsePoint)

model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)

valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds \
    .map(lambda (v, p): (v - p)**2) \
    .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))
