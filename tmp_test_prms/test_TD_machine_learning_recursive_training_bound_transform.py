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
import shutil
import numpy as np

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

def smooth_origin_input_cse(input_file, output_file, threshold):
	'''
	this function smooths original function predictions
	if Pt - Pt-1 > threshold, then Pt <- Pt-1
	'''
	df_input = pd.read_csv(input_file)
	# second col predicted
	predicted_name = list(df_input.columns.values)[1]
	origin_predict = df_input[predicted_name].tolist()
	for i in range(1,len(origin_predict)-1):
		if abs(origin_predict[i] - origin_predict[i+1]) > threshold:
			origin_predict[i+1] = origin_predict[i]

	# replace original prediction with smoothed prediction
	df_input[predicted_name] = pd.Series(np.asarray(origin_predict))
	df_input.to_csv(output_file, mode = 'w', index=False)

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


def split_csv_file(input_file='prms_input.csv', n_per=0.9, fir_output_file='prms_input1.csv', sec_output_file='prms_input2.csv', padding_line_num = 0):
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

	#  skip the padding lines
	for i in range(padding_line_num):
		fp.readline()

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

def split_csv_file_loop(input_file, loop_count, train_file_len, max_test_file_len, train_file='sub_results/prms_train.csv', test_file='sub_results/prms_test.csv'):
	'''
	this function is used to split files into train file and test file
	train file and test file together do not equal input file
	'''
	# -1 coz title
	row_num = obtain_total_row_num(input_file)-1
	fp = open(input_file,'r')
	fp1 = open(train_file,'w')
	fp2 = open(test_file,'w')
	# write title
	title = fp.readline()
	fp1.write(title)
	fp2.write(title)

	#  skip the padding lines
	for i in range(max_test_file_len*loop_count):
		fp.readline()
		
	cur_row_num = 0
	# test
	# test_line_count = 0
	while cur_row_num < (train_file_len+max_test_file_len):
		tmp_line = fp.readline()
		if cur_row_num < train_file_len:
			fp1.write(tmp_line)
		else:
			fp2.write(tmp_line)
			# test_line_count = test_line_count +1
		cur_row_num = cur_row_num + 1

	fp.close()
	fp1.close()
	fp2.close()
	# print test_line_count
	print 'Split file done....'

def collect_corresponding_obs_pred(input_df, time_list):
	'''
	this function collects corresponding values
	based on time info, and return obs and original pred
	'''
	obs_list = []
	original_pred_list = []
	for i in time_list:
		time_info = i.split('--')
		year = time_info[0]
		month = time_info[1]
		day = time_info[2]
		aim_df = input_df.query('year=='+year+' & month=='+month+' & day=='+day)
		obs_list.append(float(aim_df['runoff_obs']))
		original_pred_list.append(float(aim_df['basin_cfs_pred']))
	return obs_list, original_pred_list

def merge_bound_file(original_file, file_path,loop_time):
	'''
	this function merge bound0.csv, bound1.csv, ..., boundn-1.csv 
	and return rmse
	'''
	bound_loc = file_path+'bound.csv'
	fp = open(bound_loc,'w')
	# write header
	fp_tmp = open(file_path+'bound0.csv','r')
	fp.write(fp_tmp.readline())
	fp_tmp.close()
	for i in range(loop_time):
		fp_tmp = open(file_path+'bound'+str(i)+'.csv','r')
		# skip header
		fp_tmp.readline()
		for line in fp_tmp:
			# skip line with empty space
			if line not in ['\n', '\r\n']:
				# print line
				fp.write(line)
		fp_tmp.close()
	fp.close()

	# make sure that all values above 0, coz physical meaning
	df_delta = pd.read_csv(bound_loc)
	df_origin = pd.read_csv(original_file)
	time_list = df_delta['time'].tolist()
	truth,origin_pred = collect_corresponding_obs_pred(df_origin,time_list)

	lower_error = df_delta['lower'].tolist()
	lower = [x + y for x, y in zip(lower_error,origin_pred)]
	df_delta['lower'] = pd.Series(np.asarray(lower))

	upper_error = df_delta['upper'].tolist()
	upper = [x + y for x, y in zip(upper_error,origin_pred)]
	df_delta['upper'] = pd.Series(np.asarray(upper))

	prediction_error = df_delta['prediction'].tolist()
	prediction = [x + y for x, y in zip(prediction_error,origin_pred)]
	# need series
	df_delta['prediction'] = pd.Series(np.asarray(prediction))
	df_delta['ground_truth'] = pd.Series(np.asarray(truth))

	# replace negative values with zeros
	num = df_delta._get_numeric_data()
	num[num<0] = 0

	df_delta.to_csv(bound_loc,index=False)
	# get rmse
	return get_root_mean_squared_error(df_delta['prediction'],truth)
	# return get_root_mean_squared_error(df_delta['prediction'],df_delta['ground_truth'].tolist())

def exec_regression_by_name(train_file, test_file, regression_technique, window_per, best_alpha,app_path, best_a, best_b, recursive = True, transformation = True, max_row_num=500):
	'''
	!!!!!!!!!!!!!!!file should be ordered based on time, from oldest to latest
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	max_row_num means each spark program max handle row num
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


	# should not count header
	test_file_len = obtain_total_row_num(test_file) - 1
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


def exec_regression(filename, regression_technique, window_per, best_alpha,app_path, best_a, best_b, recursive = True, transformation = True, max_row_num=500):
	'''
	!!!!!!!!!!!!!!!file should be ordered based on time, from oldest to latest
	this function run decision tree regression
	, output the results in a log file, and return the 
	predicted delta error col
	max_row_num means each spark program max handle row num
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

	# should not count header
	test_file_len = obtain_total_row_num(test_file) - 1
	if test_file_len < max_row_num:
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
	else:
		tmp_dirt = 'sub_results/'
		loop_time = int(math.ceil(float(test_file_len)/max_row_num))
		# if folder does not exist create folder, if not delete
		if not os.path.exists(tmp_dirt):
			os.makedirs(tmp_dirt)
		else:
			shutil.rmtree(tmp_dirt)
			os.makedirs(tmp_dirt)

		train_file_len = obtain_total_row_num(train_file) - 1

		bound_loc = app_path+'/'+tmp_dirt+'bound.csv'
		# if os.path.isfile(bound_loc):
		# 	# if file exist
		# 	os.remove(bound_loc)

		for i in range(loop_time):
			tmp_test_file = tmp_dirt+'prms_test'+str(i)+'.csv'
			tmp_train_file= tmp_dirt+'prms_train'+str(i)+'.csv'
			# split files into train and test
			split_csv_file_loop(filename, i, train_file_len, max_row_num, tmp_train_file, tmp_test_file)
			# print 'current max_row_num: '+str(max_row_num)+'; current loop num: '+str(loop_time)
			# break
			delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
			delta_error_filename_train = app_path + '/delta_error_train.libsvm'
			delta_error_file(tmp_train_file,delta_error_csv_train,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

			# test file
			delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
			delta_error_filename_test = app_path + '/delta_error_test.libsvm'
			delta_error_file(tmp_test_file,delta_error_csv_test,best_alpha)
			convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)

			if recursive == True:
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path+'/'+tmp_dirt.replace('/',''), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
			else:
				command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
			with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
				process = subprocess.Popen(
					command, stdout=process_out, stderr=err_out, cwd=app_path)

			# this waits the process finishes
			process.wait()
			print str(i)+'th part of the file is processing'
			print str(loop_time-i-1)+'left for processed'

			if os.path.isfile(bound_loc):
				# if file exist
				shutil.copyfile(bound_loc,app_path+'/'+tmp_dirt+'bound'+str(i)+'.csv')
			# TODO need to merge result file too!!!!

		print 'final rmse is: '+str(merge_bound_file(filename, app_path+'/'+tmp_dirt,loop_time))

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
		alpha = 0.1*(alpha_count+1)
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

		for a_count in range(10):
		# change!!!!!!!!!!!!!!!!!!!!
		# for a_count in range(1):
			tmp_a = 0.01*(a_count+1)+0.0005

			for b_count in range(10):
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

	exec_regression(filename, regression_technique, window_per, best_alpha,app_path, best_a, best_b, True, True, 500)

	# # training file
	# delta_error_csv_train = app_path + '/temp_delta_error_train.csv'
	# delta_error_filename_train = app_path + '/delta_error_train.libsvm'
	# # observed_name, predicted_name = delta_error_file(filename,delta_error_filename)
	# delta_error_file(train_file,delta_error_csv_train,best_alpha)
	# convert_csv_into_libsvm(delta_error_csv_train,delta_error_filename_train)

	# # test file
	# delta_error_csv_test = app_path + '/temp_delta_error_test.csv'
	# delta_error_filename_test = app_path + '/delta_error_test.libsvm'
	# # observed_name, predicted_name = dereal_crossover_exec_regression('smoothed_prms_input.csv','gb_tree',0.5)lta_error_file(filename,delta_error_filename)
	# delta_error_file(test_file,delta_error_csv_test,best_alpha)
	# convert_csv_into_libsvm(delta_error_csv_test,delta_error_filename_test)

	# command = [spark_submit_location, exec_file_loc,delta_error_filename_train,result_file, str(window_per), str(best_alpha), app_path, str(best_a), str(best_b), delta_error_filename_test, spark_config1, spark_config2]
	# with open(log_path, 'wb') as process_out, open(log_path, 'rb', 1) as reader, open(err_log_path, 'wb') as err_out:
	# 	process = subprocess.Popen(
	# 		command, stdout=process_out, stderr=err_out, cwd=app_path)

	# # this waits the process finishes
	# process.wait()
	# cur_avg_rmse = get_avg(result_file)
	# print "final rmse is: "+str(cur_avg_rmse)

	return True




# input file, first column is observation, second column is prediction
# exec_regression('prms_input.csv','decision_tree')


# print 'original rmse is: '+str(original_csv_rmse('prms_input.csv',0.9))

# exec_regression('prms_input.csv', 'gb_tree', 0.9, 0.8,app_path, 0.0105, 0.0205, True, True, 100)

# exec_regression_by_name('sub_results/prms_train0.csv', 'sub_results/prms_test0.csv', 'gb_tree', 0.2, 0.4,app_path, 0.0405, 0.0505)
# print original_csv_rmse('prms_input.csv', window_per=0.4)

#merge_bound_file('smoothed_prms_input.csv', file_path,loop_time)

# create smooth version input file
# smooth_origin_input_cse('prms_input.csv', 'smoothed_prms_input.csv', 30)
#real_crossover_exec_regression('smoothed_prms_input.csv','gb_tree',0.5)
#exec_regression('smoothed_prms_input.csv', 'gb_tree',0.5, 0.6,app_path, 0.0305, 0.0105, True, True, 500)

# smooth_origin_input_cse('prms_input.csv', 'smoothed_prms_input.csv', 10)
# exec_regression('smoothed_prms_input.csv', 'gb_tree',0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 500)


smooth_origin_input_cse('prms_input_without_calibrate.csv', 'smoothed_prms_input.csv', 10)
real_crossover_exec_regression('smoothed_prms_input.csv','gb_tree',0.5)
# exec_regression('smoothed_prms_input.csv', 'gb_tree',0.5, 0.9,app_path, 0.1005, 0.0705, True, True, 500)