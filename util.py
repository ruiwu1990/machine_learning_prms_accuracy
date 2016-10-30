import pandas as pd
import math
import subprocess
import sys
import csv
import os
from collections import defaultdict
import json

app_path = os.path.dirname(os.path.abspath(__file__))

def get_predict_observed(filename,predicted_name,observed_name):
	'''
	the function uses pandas to get the
	predicted and observed
	return a list like this [(predict0, observed0),(predict1, observed1),...]
	'''
	csv_file = pd.read_csv(filename)
	predicted = csv_file[predicted_name]
	observed = csv_file[observed_name]
	predicted_list = predicted.tolist()
	observed_list = observed.tolist()
	return [list(a) for a in zip(predicted_list, observed_list)]

def get_real_delta_error(filename,predicted_name,observed_name):
	'''
	the function returns the real delta error
	'''
	csv_file = pd.read_csv(filename)
	predicted = csv_file[predicted_name]
	observed = csv_file[observed_name]
	predicted_list = predicted.tolist()
	observed_list = observed.tolist()
	return [observed_list_i - predicted_list_i for observed_list_i, predicted_list_i in zip(observed_list, predicted_list)]

def add_delta_error_prediced(e_list,p_o_list):
	'''
	This function add error into predicted value
	p_o_list is from function get_predict_observed
	'''
	if len(e_list) != len(p_o_list):
		raise Exception('two lists have different lengths')
	list_len = len(e_list)
	for count in range(list_len):
		p_o_list[count][0] = p_o_list[count][0] + e_list[count]



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

def get_delta_error_col(spark_df,e_col_name):
	'''
	this function get the delta error col
	'''
	pd_df = spark_df.toPandas()
	return pd_df[e_col_name].tolist()

def delta_error_file(filename, e_filename):
	'''
	this function replace the first column (observed)
	and second column (predicted values) with delta_e
	'''
	pd_df = pd.read_csv(filename)
	# add the delta error col
	# first col observed
	observed_name = list(pd_df.columns.values)[0]
	# second col predicted
	predicted_name = list(pd_df.columns.values)[1]
	pd_df.insert(0,'delta_e',pd_df[observed_name].sub(pd_df[predicted_name]))
	# remove observed and predicted col
	del pd_df[observed_name]
	del pd_df[predicted_name]
	pd_df.to_csv(e_filename,index=False)
	return observed_name, predicted_name

def get_delta_e_GBT(filename):
	'''
	this function train the GLR
	model and return the three columns list, predicted error,
	observed, model predicted data
	'''
	# create delta error file
	delta_error_filename = app_path + '/static/data/delta_error.csv'
	observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	exec_GBT_regression(delta_error_filename)
	# results are stored here
	result_file = app_path + '/gbt_result.txt'
	fp = open(result_file, 'r')
	result_accuracy = fp.readline()
	predicted_delta_e = [float(i) for i in fp.readline().strip().split(',')]
	fp.close()
	# get the [predicted_name , observed_name] as p_o_list
	p_o_list = get_predict_observed(filename,predicted_name,observed_name)
	temp=zip(*[(a,b) for a,b in p_o_list])
	original_p_list = list(temp[0])
	o_list = list(temp[1])
	# add delta_e into model predicted values
	add_delta_error_prediced(predicted_delta_e,p_o_list)
	temp=zip(*[(a,b) for a,b in p_o_list])
	improved_p_list = list(temp[0])

	original_rmse = get_root_mean_squared_error(original_p_list,o_list)
	improved_rmse = get_root_mean_squared_error(improved_p_list,o_list)

	original_error = "the original rmse is:" + str(original_rmse)
	improved_error = "the improved rmse is:" + str(improved_rmse)

	return json.dumps({'accuracy_info':result_accuracy,'original_p_list':original_p_list,'improved_p_list':improved_p_list,'o_list':o_list,'original_error':original_error,'improved_error':improved_error})

def get_delta_e_GLR(filename):
	'''
	this function train the GLR
	model and return the three columns list, predicted error,
	observed, model predicted data
	'''
	# create delta error file
	delta_error_filename = app_path + '/static/data/delta_error.csv'
	observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	exec_GLR_regression(delta_error_filename)
	# results are stored here
	result_file = app_path + '/glr_result.txt'
	fp = open(result_file, 'r')
	result_accuracy = fp.readline()
	predicted_delta_e = [float(i) for i in fp.readline().strip().split(',')]
	fp.close()
	# get the [predicted_name , observed_name] as p_o_list
	p_o_list = get_predict_observed(filename,predicted_name,observed_name)
	temp=zip(*[(a,b) for a,b in p_o_list])
	original_p_list = list(temp[0])
	o_list = list(temp[1])
	# add delta_e into model predicted values
	add_delta_error_prediced(predicted_delta_e,p_o_list)
	temp=zip(*[(a,b) for a,b in p_o_list])
	improved_p_list = list(temp[0])

	original_rmse = get_root_mean_squared_error(original_p_list,o_list)
	improved_rmse = get_root_mean_squared_error(improved_p_list,o_list)

	original_error = "the original rmse is:" + str(original_rmse)
	improved_error = "the improved rmse is:" + str(improved_rmse)

	return json.dumps({'accuracy_info':result_accuracy,'original_p_list':original_p_list,'improved_p_list':improved_p_list,'o_list':o_list,'original_error':original_error,'improved_error':improved_error})


def get_delta_e_RF(filename):
	'''
	this function train the decision tree regression
	model and return the three columns list, predicted error,
	observed, model predicted data
	'''
	# create delta error file
	delta_error_filename = app_path + '/static/data/delta_error.csv'
	observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	exec_RF_regression(delta_error_filename)
	# results are stored here
	result_file = app_path + '/rf_result.txt'
	fp = open(result_file, 'r')
	result_accuracy = fp.readline()
	predicted_delta_e = [float(i) for i in fp.readline().strip().split(',')]
	fp.close()
	# get the [predicted_name , observed_name] as p_o_list
	p_o_list = get_predict_observed(filename,predicted_name,observed_name)
	temp=zip(*[(a,b) for a,b in p_o_list])
	original_p_list = list(temp[0])
	o_list = list(temp[1])
	# add delta_e into model predicted values
	add_delta_error_prediced(predicted_delta_e,p_o_list)
	temp=zip(*[(a,b) for a,b in p_o_list])
	improved_p_list = list(temp[0])

	original_rmse = get_root_mean_squared_error(original_p_list,o_list)
	improved_rmse = get_root_mean_squared_error(improved_p_list,o_list)

	original_error = "the original rmse is:" + str(original_rmse)
	improved_error = "the improved rmse is:" + str(improved_rmse)

	return json.dumps({'accuracy_info':result_accuracy,'original_p_list':original_p_list,'improved_p_list':improved_p_list,'o_list':o_list,'original_error':original_error,'improved_error':improved_error})
	

def get_delta_e_decision_tree(filename):
	'''
	this function train the decision tree regression
	model and return the three columns list, predicted error,
	observed, model predicted data
	'''
	# create delta error file
	delta_error_filename = app_path + '/static/data/delta_error.csv'
	observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	exec_decision_tree_regression(delta_error_filename)
	# results are stored here
	result_file = app_path + '/decision_tree_result.txt'
	fp = open(result_file, 'r')
	result_accuracy = fp.readline()
	predicted_delta_e = [float(i) for i in fp.readline().strip().split(',')]
	fp.close()
	# get the [predicted_name , observed_name] as p_o_list
	p_o_list = get_predict_observed(filename,predicted_name,observed_name)
	temp=zip(*[(a,b) for a,b in p_o_list])
	original_p_list = list(temp[0])
	o_list = list(temp[1])

	real_delta_error = get_real_delta_error(filename,predicted_name,observed_name)
	# add delta_e into model predicted values
	add_delta_error_prediced(predicted_delta_e,p_o_list)
	temp=zip(*[(a,b) for a,b in p_o_list])
	improved_p_list = list(temp[0])

	original_pbias = get_pbias(original_p_list,o_list)
	improved_pbias = get_pbias(improved_p_list,o_list)
	delta_error_pbias = get_pbias(predicted_delta_e,real_delta_error)

	original_cd = get_coeficient_determination(original_p_list,o_list)
	improved_cd = get_coeficient_determination(improved_p_list,o_list)
	delta_error_cd = get_coeficient_determination(predicted_delta_e,real_delta_error)

	original_nse = get_nse(original_p_list,o_list)
	improved_nse = get_nse(improved_p_list,o_list)
	delta_error_nse = get_nse(predicted_delta_e,real_delta_error)

	original_rmse = get_root_mean_squared_error(original_p_list,o_list)
	improved_rmse = get_root_mean_squared_error(improved_p_list,o_list)
	delta_error_rmse = get_root_mean_squared_error(predicted_delta_e,real_delta_error)

	original_rmse_error = "the original rmse is:" + str(original_rmse)
	improved_rmse_error = "the improved rmse is:" + str(improved_rmse)

	return json.dumps({'original_p_list':original_p_list,\
					   'improved_p_list':improved_p_list,\
					   'o_list':o_list,\
					   'original_pbias':original_pbias, \
					   'improved_pbias':improved_pbias, \
					   'delta_error_pbias':delta_error_pbias,\
					   'original_cd':original_cd, \
					   'improved_cd':improved_cd, \
					   'delta_error_cd':delta_error_cd,\
					   'original_nse':original_nse, \
					   'improved_nse':improved_nse, \
					   'delta_error_nse':delta_error_nse,\
					   'accuracy_info':result_accuracy,\
					   'original_rmse':original_rmse,\
					   'improved_rmse':improved_rmse,\
					   'delta_error_rmse':delta_error_rmse})
	

# the following construct_line and convert_csv_into_libsvm
# convert csv into libsvm
# the function is basically 
# from https://github.com/zygmuntz/phraug/blob/master/csv2libsvm.py
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

def exec_RF_regression(filename):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	# get the libsvm file
	output_file = app_path + '/static/data/delta_error.libsvm'
	convert_csv_into_libsvm(filename,output_file)
	log_path = app_path + '/rf_log.txt'
	err_log_path = app_path + '/rf_err_log.txt'
	exec_file_loc = app_path + '/ml_moduel/random_forest_regression.py'
	result_file = app_path + '/rf_result.txt'
	command = ['spark-submit',exec_file_loc,output_file,result_file]
	# execute the model
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	return True

def exec_GBT_regression(filename):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	# get the libsvm file
	output_file = app_path + '/static/data/delta_error.libsvm'
	convert_csv_into_libsvm(filename,output_file)
	log_path = app_path + '/gbt_log.txt'
	err_log_path = app_path + '/gbt_err_log.txt'
	exec_file_loc = app_path + '/ml_moduel/gradient_boosted_regression.py'
	result_file = app_path + '/gbt_result.txt'
	command = ['spark-submit',exec_file_loc,output_file,result_file]
	# execute the model
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	return True

def exec_GLR_regression(filename):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	# get the libsvm file
	output_file = app_path + '/static/data/delta_error.libsvm'
	convert_csv_into_libsvm(filename,output_file)
	log_path = app_path + '/glr_log.txt'
	err_log_path = app_path + '/glr_err_log.txt'
	exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
	result_file = app_path + '/glr_result.txt'
	command = ['spark-submit',exec_file_loc,output_file,result_file]
	# execute the model
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	return True


def exec_decision_tree_regression(filename):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	# get the libsvm file
	output_file = app_path + '/static/data/delta_error.libsvm'
	convert_csv_into_libsvm(filename,output_file)
	log_path = app_path + '/decision_tree_log.txt'
	err_log_path = app_path + '/decision_tree_err_log.txt'
	exec_file_loc = app_path + '/ml_moduel/decision_tree_regression.py'
	result_file = app_path + '/decision_tree_result.txt'
	command = ['spark-submit',exec_file_loc,output_file,result_file]
	# execute the model
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	return True

