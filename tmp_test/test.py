import pandas as pd
import math
from random import randint

pi = 3.14159265358979323846264

def calculate_dt(filename):
	'''
	this function calculates dtime and store results
	in an array
	'''
	# fp = open('a.csv','w')
	df = pd.read_csv(filename)
	row_len = len(df)
	result_dt = []
	for r in range(row_len):
		dec_time1 = float(df['month'][r]*30 + df['day'][r])/365.0
		dec_time2 = float(df['hour'][r]*60+df['minute'][r])/(365.0*24*60)
		temp = dec_time1 + dec_time2 + df['year'][r]
		result_dt.append(temp)
		# fp.write(str(temp)+'\n')

	# fp.close()
	center = sum(result_dt)/row_len
	for r in range(row_len):
		result_dt[r] = result_dt[r] - center

	return result_dt


def create_input_feature_file(filename,output_file):
	'''
	nitrogen input files cannot be used directly
	coz the features need to be transferred
	the gound truth and prediction are in the last two cols
	'''
	df = pd.read_csv(filename)
	row_len = len(df)
	dt = calculate_dt(filename)
	fp = open(output_file,'w')
	# header
	fp.write('0,1,2,3,4,5,6,7,8,9,10,11,12,13\n')
	for r in range(row_len):
		temp = []
		temp.append('1')
		# some predicted Q are negative
		Q = 1e-100
		if df['Q'][r] > 0:
			Q = df['Q'][r]
		temp.append(str(math.log(Q)))
		temp.append(str(math.log(Q)*math.log(Q)))
		temp.append(str(math.sin(2*pi*dt[r])))
		temp.append(str(math.cos(2*pi*dt[r])))
		temp.append(str(dt[r]))
		temp.append(str(dt[r]*dt[r]))
		temp.append(str(df['DO'][r]))
		temp.append(str(df['T'][r]))
		temp.append(str(df['ON'][r]))
		temp.append(str(df['OP'][r]))
		temp.append(str(df['TP'][r]))
		temp.append(str(df['ground_truth'][r]))
		temp.append(str(df['GA'][r]))
		# print temp
		fp.write(','.join(temp)+'\n')

	fp.close()




def ordered_td_prediction(filename, train_per, test_per,init_w, alpha = 1):
	'''
	td prediction
	first 70% training and left 30% testing
	the function is for linear features
	'''
	df = pd.read_csv(filename)
	row_len = len(df)
	train_row = int(row_len*train_per)
	test_row = row_len - train_row
	w = init_w
	# random pick a continuous trunck from the data
	start = randint(0,train_row-1)
	error_squre = 0
	new_pred = []
	for r in range(test_row):
		temp_result = 0
		for count in range(len(w)):
			temp_result = temp_result + w[count]*df[str(count)][r]
		new_pred.append(temp_result)
		error_squre = error_squre + (temp_result-df[str(len(w)-1)][r])*(temp_result-df[str(len(w)-2)][r])
		# update w
		factor = alpha*(df[str(len(w)-2)][start+r] - df[str(len(w)-1)][start+r])
		for c in range(len(w)):
			w[c] = w[c] + factor*w[c]

	print "rmse is: "+str(math.sqrt(error_squre)/test_row)



# a = calculate_dt('data.csv')
create_input_feature_file('data.csv','feature.csv')

w = [1.26,-6.06,3.15,0.72,0.43,-0.17,2.95,-0.02,0,1.28,6.61,-5.55]
ordered_td_prediction('data.csv',0.7,0.3,w)