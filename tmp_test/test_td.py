import pandas as pd
import math
from random import randint
import random
import copy

pi = 3.14159265359

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
		Q = 0.00001
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




def ordered_td_prediction(filename, train_per, test_per,init_w, alpha = 1, crossover_times=30):
	'''
	td prediction
	first 70% training and left 30% testing
	the function is for linear features
	'''
	df = pd.read_csv(filename)
	row_len = len(df)
	train_row = int(row_len*train_per)
	test_row = row_len - train_row
	rmse = 0
	new_pred = []
	ground_truth = []
	for c_count in range(crossover_times):
		chosen_list = random.sample(range(row_len),test_row)
		w = copy.deepcopy(init_w)
		# test
		# chosen_list = range(test_row)
		# training phase
		training_list = [x for x in range(row_len) if x not in chosen_list]
		# print training_list
		# new_pred = []
		# ground_truth = []
		for t in range(train_row):
			temp_result1 = 0
			# test
			for count in range(len(w)):
				temp_result1 = temp_result1 + w[count]*df[str(count)][training_list[t]]
			# new_pred.append(temp_result1)
			# update w
			factor = alpha*(df[str(len(w))][training_list[t]] - df[str(len(w)+1)][training_list[t]])
			# ground_truth.append(df[str(len(w)-2)][training_list[t]])
			
			for c in range(len(w)):
				w[c] = w[c] + factor*w[c]
		# if c_count == 0 or c_count == crossover_times-1:
		# 	# print new_pred
		# 	print test_row
		# 	print ground_truth
		new_pred = []
		ground_truth = []
		# testing phase
		for r in range(test_row):
			temp_result = 0
			# test
			# print "----------------------"
			for count in range(len(w)):
				temp_result = temp_result + w[count]*df[str(count)][chosen_list[r]]
			new_pred.append(temp_result)
			# update w
			factor = alpha*(df[str(len(w))][chosen_list[r]] - df[str(len(w)+1)][chosen_list[r]])
			ground_truth.append(df[str(len(w))][chosen_list[r]])
			for c in range(len(w)):
				w[c] = w[c] + factor*w[c]

		# if c_count == 0 or c_count == crossover_times-1:
		# 	# print new_pred
		# 	print ground_truth
		# print "current rmse: "+str(get_root_mean_squared_error(new_pred,ground_truth))+"; current rmse sum is: "+str(rmse)
		rmse = rmse + get_root_mean_squared_error(new_pred,ground_truth)

	rmse = rmse/crossover_times
	print "rmse is: "+str(rmse)
	return rmse, new_pred, ground_truth

# ------------------------
# test matrix functions

def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

def get_pbias(list1, list2):
	'''
	percent bias
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	sum_original = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])
		sum_original = sum_original + list2[count]
	result = sum_diff/sum_original
	return result*100

def get_coeficient_determination(list1,list2):
	'''
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	mean_list1 = reduce(lambda x, y: x + y, list1) / len(list1)
	mean_list2 = reduce(lambda x, y: x + y, list2) / len(list2)
	sum_diff = 0
	sum_diff_o_s = 0
	sum_diff_p_s = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-mean_list1)*(list2[count]-mean_list2)
		sum_diff_o_s = sum_diff_o_s + (list2[count]-mean_list2)**2
		sum_diff_p_s = sum_diff_p_s + (list1[count]-mean_list1)**2
	result = (sum_diff/(pow(sum_diff_o_s,0.5)*pow(sum_diff_p_s,0.5)))**2
	return result

def get_nse(list1,list2):
	'''
	Nash-Sutcliffe efficiency
	list1 is model simulated value
	list2 is observed data
	'''
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff_power = 0
	sum_diff_o_power = 0
	mean_list2 = reduce(lambda x, y: x + y, list2) / len(list2)
	for count in range(list_len):
		sum_diff_power = sum_diff_power + (list1[count]-list2[count])**2
		sum_diff_o_power = sum_diff_o_power + (list2[count]-mean_list2)**2
	result = sum_diff_power/sum_diff_o_power
	return 1 - result

# ------------------------

# a = calculate_dt('data.csv')
create_input_feature_file('data.csv','feature.csv')

w = [1.26,-6.06,3.15,0.72,0.43,-0.17,2.95,-0.02,0,1.28,6.61,-5.55]

# ordered_td_prediction('feature.csv',0.7,0.3,w,0.8)


min_rmse = 100
alpha = -1
for i in range(1,20):
	w = [1.26,-6.06,3.15,0.72,0.43,-0.17,2.95,-0.02,0,1.28,6.61,-5.55]
	print "current alpha is: "+str(0.05*i)
	temp_rmse,a,b = ordered_td_prediction('feature.csv',0.7,0.3,w,0.05*i)
	if temp_rmse <= min_rmse:
		min_rmse = temp_rmse
		alpha = 0.05*i

print "best alpha is: "+str(alpha)+"; best rmse is: "+str(min_rmse)
temp_rmse,a,b = ordered_td_prediction('feature.csv',0.7,0.3,w,alpha)
print "pbias: "+str(get_pbias(a,b))+"; cd is: "+str(get_coeficient_determination(a,b))+"; nse is: "+str(get_nse(a,b))
