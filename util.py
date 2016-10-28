import pandas as pd
import math
import subprocess
import sys
import csv
import os
from collections import defaultdict

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

# separate p_o_list into 2 list
'''
temp=zip(*[(a,b) for a,b in p_o_list])
p_list = list(temp[0])
o_list = list(temp[1])
'''


def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)

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
	pd_df.to_csv(e_filename)

def get_delta_e_decision_tree(filename):
	'''
	this function train the decision tree regression
	model and return the three columns list, predicted error,
	observed, model predicted data
	'''
	# create delta error file
	delta_error_filename = app_path + '/static/data/delta_error.csv'
	delta_error_file(filename,delta_error_filename)
	exec_decision_tree_regression(delta_error_filename)

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

def exec_decision_tree_regression(filename):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	# get the libsvm file
	# test
	print 'jose is panda:'+filename
	output_file = app_path + '/static/data/delta_e.libsvm'
	convert_csv_into_libsvm(filename,output_file)
	log_path = app_path + '/decision_tree_log.txt'
	err_log_path = app_path + '/decision_tree_err_log.txt'
	exec_file_loc = app_path + '/ml_moduel/decision_tree_regression.py'
	command = ['spark-submit',exec_file_loc,filename]
	# execute the model
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	return True

