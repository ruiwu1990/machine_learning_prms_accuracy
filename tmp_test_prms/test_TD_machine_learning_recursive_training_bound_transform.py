'''
This file use machine learning to find the direction
and use modified TD to improve model accuracy
I may need to change the timeout configuration by using this:
/cse/home/rwu/Desktop/hadoop/spark_installation/spark-2.1.0/bin/pyspark --conf spark.executor.heartbeatInterval=10000000 --conf spark.network.timeout=10000000
'''
import pandas as pd
import math
import subprocess
import sys
import csv
import os
from collections import defaultdict
import json
import os

app_path = os.path.dirname(os.path.abspath('__file__'))
spark_submit_location = '/home/host0/Desktop/hadoop/spark-2.1.0/bin/spark-submit'
spark_config1 = '--conf spark.executor.heartbeatInterval=10000000'
spark_config2 = '--conf spark.network.timeout=10000000'
# this will be '/cse/home/rwu/Desktop/machine_learning_prms_accuracy/tmp_test'
# /cse/home/rwu/Desktop/hadoop/spark_installation/spark-2.1.0/bin/pyspark

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

def delta_error_file(filename, e_filename, alpha=1):
	'''
	this function replace the first column (observed)
	and second column (predicted values) with delta_e
	alpha is factor for final_delta = alpha*delta
	'''
	pd_df = pd.read_csv(filename)
	# add the delta error col
	# first col observed
	observed_name = list(pd_df.columns.values)[0]
	# second col predicted
	predicted_name = list(pd_df.columns.values)[1]
	pd_df.insert(0,'delta_e',alpha*pd_df[observed_name].sub(pd_df[predicted_name]))
	# remove observed and predicted col
	del pd_df[observed_name]
	del pd_df[predicted_name]
	pd_df.to_csv(e_filename,index=False)
	return observed_name, predicted_name

def get_avg(filename):
	'''
	this function get avg value of a single col file
	'''
	fp = open(filename, 'r')
	sum_result = 0
	count = 0
	for line in fp:
		count = count + 1
		sum_result = float(line) + sum_result
	fp.close()
	return sum_result/count

def obtain_total_row_num(filename):
	'''
	get file total lines
	'''
	fp = open(filename,'r')
	result = sum(1 for row in fp)
	fp.close()
	return result

def split_csv_file(input_file='prms_input.csv', n_per=0.9, fir_output_file='prms_input1.csv', sec_output_file='prms_input2.csv'):
	'''
	fir_output_file n_per and sec_output_file 1-n_per
	'''
	# -1 coz title
	row_num = obtain_total_row_num(input_file)-1
	fir_row = int(row_num*n_per)
	fp = open(input_file,'r')
	fp1 = open(fir_output_file,'w')
	fp2 = open(sec_output_file,'w')
	# write title
	title = fp.readline()
	fp1.write(title)
	fp2.write(title)

	cur_row_num = 0
	while cur_row_num < row_num:
		tmp_line = fp.readline()
		if cur_row_num < fir_row:
			fp1.write(tmp_line)
		else:
			fp2.write(tmp_line)
		cur_row_num = cur_row_num + 1

	fp.close()
	fp1.close()
	fp2.close()

def exec_regression(filename, regression_technique, window_per, best_alpha,app_path, best_a, best_b, recursive = True, transformation = True):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	if regression_technique =='rf':
		log_path = app_path + '/rf_log.txt'
		err_log_path = app_path + '/rf_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/random_forest_regression.py'
		result_file = app_path + '/rf_result.txt'

	elif regression_technique =='decision_tree':
		log_path = app_path + '/decision_tree_log.txt'
		err_log_path = app_path + '/decision_tree_err_log.txt'
		# change!!!!!!!!!!!!!!!
		if recursive == True and transformation == True:
			exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_log_sinh.py'
		elif recursive == False and transformation == True:
			exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_logsinh_final_test.py'
		elif recursive == True and transformation == False:
			exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_no_transform_interval_log_sinh.py'
		elif recursive == False and transformation == False:
			exec_file_loc = app_path + '/ml_moduel/decision_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		result_file = app_path + '/decision_tree_result.txt'

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
		result_file = app_path + '/glr_result.txt'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		if recursive == True and transformation == True:
			exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_interval_log_sinh.py'
		elif recursive == False and transformation == True:
			exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_transform_no_recursive_logsinh_final_test.py'
		elif recursive == True and transformation == False:
			exec_file_loc = app_path + '/ml_moduel/td_gb_tree_regression_prediction_no_transform_interval_log_sinh.py'
		elif recursive == False and transformation == False:
			exec_file_loc = app_path + '/ml_moduel/gb_tree_regression_no_transform_no_recursive_logsinh_final_test.py'
		result_file = app_path + '/gbt_result.txt'
	else:
		print 'Sorry, current system does not support the input regression technique'
	
	if os.path.isfile(result_file):
		# if file exist
		os.remove(result_file)

	train_file='prms_input1.csv'
	test_file='prms_input2.csv'
	split_csv_file(filename, window_per, train_file, test_file)
	# training file
	delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
	delta_error_filename_train = app_path + '/delta_error_train.libsvm'
	# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	delta_error_file(train_file,delta_error_csv_train,best_alpha)
	convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

	# test file
	delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
	delta_error_filename_test = app_path + '/delta_error_test.libsvm'
	# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	delta_error_file(test_file,delta_error_csv_test,best_alpha)
	convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)
	if recursive == True:
		command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
	else:
		command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	cur_avg_rmse = get_avg(result_file)
	print "final rmse is: "+str(cur_avg_rmse)
	return True



def real_crossover_exec_regression(filename, regression_technique, window_per=0.9, training_window_per = 0.9):
	'''
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	'''
	if regression_technique =='rf':
		log_path = app_path + '/rf_log.txt'
		err_log_path = app_path + '/rf_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/random_forest_regression.py'
		result_file = app_path + '/rf_result.txt'

	elif regression_technique =='decision_tree':
		log_path = app_path + '/decision_tree_log.txt'
		err_log_path = app_path + '/decision_tree_err_log.txt'
		# change!!!!!!!!!!!!!!!
		exec_no_recursive_file_loc = app_path + '/ml_moduel/decision_tree_regression_transform_no_recursive_logsinh.py'
		exec_file_loc = app_path + '/ml_moduel/td_decision_tree_regression_prediction_interval_log_sinh.py'
		# exec_file_loc = app_path + '/ml_moduel/decision_tree_regression.py'
		result_file = app_path + '/decision_tree_result.txt'

	elif regression_technique =='glr':
		log_path = app_path + '/glr_log.txt'
		err_log_path = app_path + '/glr_err_log.txt'
		exec_file_loc = app_path + '/ml_moduel/generalized_linear_regression.py'
		result_file = app_path + '/glr_result.txt'

	elif regression_technique =='gb_tree':
		log_path = app_path + '/gbt_log.txt'
		err_log_path = app_path + '/gbt_err_log.txt'
		exec_no_recursive_file_loc = app_path + '/ml_moduel/gb_tree_regression_transform_no_recursive_logsinh.py'
		exec_file_loc = app_path + '/ml_moduel/td_gd_tree_regression_prediction_interval_log_sinh.py'
		result_file = app_path + '/gbt_result.txt'
	else:
		print 'Sorry, current system does not support the input regression technique'
	
	min_rmse = 10000
	best_alpha = -1
	fp1 = open('all_results.csv','w')
	# # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+exec_file_loc
	# test_cases = [[0.3,0.8],[0.8,0.9],[0.8,0.7],[0.4,0.9],[0.1,0.9],[0.9,0.9],[0.7,0.6],[0.6,0.9],[0.1,0.8],[0.2,0.7]]
	# # change!!!!!!!!!!!!!!!
	# # for window_count in range(9):
	# for window_count in range(1):
	# 	# max window size is 95%
	# 	# window_per = 0.1*(window_count+1)
	# 	# change!!!!!!!!!!!!!!!
	# 	# window_per = test_cases[window_count][1]
	# 	window_per = 0.9
	train_file='prms_input1.csv'
	test_file='prms_input2.csv'
	split_csv_file(filename, window_per, train_file, test_file)

	best_a = -1
	best_b = -1
	# change!!!!!!!!!!!!!!!
	for alpha_count in range(5):
	# for alpha_count in range(1):
		alpha = 0.2*(alpha_count+1)
		# change!!!!!!!!!!!!!!!
		# alpha = test_cases[window_count][0]
		# alpha = 0.3
		
		# clean previous generated results
		if os.path.isfile(result_file):
			# if file exist
			os.remove(result_file)

		# get the libsvm file
		delta_error_csv = app_path + '/temp_delta_error.csv'
		delta_error_filename = app_path + '/delta_error.libsvm'
		
		delta_error_file(train_file,delta_error_csv,alpha)
		convert_csv_into_libsvm(delta_error_csv,delta_error_filename)

		for a_count in range(5):
		# change!!!!!!!!!!!!!!!!!!!!
		# for a_count in range(1):
			tmp_a = 0.01*(a_count+1)+0.0005

			for b_count in range(5):
			# change!!!!!!!!!!!!!!!!!!!!11
			# for b_count in range(1):
				tmp_b = 0.01*(b_count+1)+0.0005

				# this command will work if source the spark-submit correctly
				# no recursive for crossover validation
				command = [spark_submit_location, exec_no_recursive_file_loc,delta_error_filename,result_file, str(training_window_per), str(alpha), str(tmp_a), str(tmp_b), spark_config1, spark_config2]
				#  30 times crossover validation
				# for i in range(30):
				# !!!!!!!!!!!!!!!!!!!change
				for i in range(10):
				# execute the model
					with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
						process = subprocess.Popen(
							command, stdout=process_out, stderr=err_out, cwd=app_path)

					# this waits the process finishes
					process.wait()
					print "current processing loop for alaph "+str(alpha)+", a: "+str(tmp_a)+", b: "+str(tmp_b)+", and crossover time: "\
							+str(i)+"//////////////////////////////"
					# sys.exit()

				cur_avg_rmse = get_avg(result_file)
				os.remove(result_file)

				print "~~~~~current avg is rmse: "+str(cur_avg_rmse)
				fp1.write(str(alpha)+","+str(window_per)+","+str(cur_avg_rmse)+","+str(tmp_a)+","+str(tmp_b)+'\n')
				if cur_avg_rmse < min_rmse:
					min_rmse = cur_avg_rmse
					best_alpha = alpha
					best_a = tmp_a
					best_b = tmp_b
	fp1.close()
	print "min rmse is: "+ str(min_rmse)+"; best alpha is: "+str(best_alpha)+"; best a is: "+str(best_a)+"; best b is: "+str(best_b)+"; current window size is: "+str(window_per)
	
	# recursive test file, with best alpha, a and b
	if os.path.isfile(result_file):
		# if file exist
		os.remove(result_file)

	# training file
	delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
	delta_error_filename_train = app_path + '/delta_error_train.libsvm'
	# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	delta_error_file(train_file,delta_error_csv_train,best_alpha)
	convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

	# test file
	delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
	delta_error_filename_test = app_path + '/delta_error_test.libsvm'
	# observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	delta_error_file(test_file,delta_error_csv_test,best_alpha)
	convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)

	command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
	with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
		process = subprocess.Popen(
			command, stdout=process_out, stderr=err_out, cwd=app_path)

	# this waits the process finishes
	process.wait()
	cur_avg_rmse = get_avg(result_file)
	print "final rmse is: "+str(cur_avg_rmse)

	return True

def get_root_mean_squared_error(list1,list2):
	if len(list1) != len(list2):
		raise Exception('two lists have different lengths')
	list_len = len(list1)
	sum_diff = 0
	for count in range(list_len):
		sum_diff = sum_diff + (list1[count]-list2[count])**2
	avg_sum_diff = sum_diff/list_len
	return math.sqrt(avg_sum_diff)


def original_csv_rmse(filename, window_per=0.9):
	'''
	this function finds original model (1-window_per%) rmse
	'''
	fir_output_file='prms_input1.csv'
	sec_output_file='prms_input2.csv'
	split_csv_file(filename, window_per, fir_output_file, sec_output_file)
	df = pd.read_csv(sec_output_file)
	return get_root_mean_squared_error(df['runoff_obs'].tolist(),df['basin_cfs_pred'].tolist())


# input file, first column is observation, second column is prediction
# exec_regression('prms_input.csv','decision_tree')
real_crossover_exec_regression('prms_input.csv','gb_tree',0.7)
# print 'original rmse is: '+str(original_csv_rmse('prms_input.csv',0.9))

# exec_regression('prms_input.csv', 'gb_tree', 0.95, 1,app_path, 0.0105, 0.0205, False, True)
# exec_regression('prms_input.csv', 'decision_tree', window_per=0.9, best_alpha=0.2,app_path, best_a=0.0205, best_b=0.0305)

