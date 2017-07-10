from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import numpy as np
from datetime import datetime
import os
from util import get_root_mean_squared_error, get_pbias, get_coeficient_determination, get_nse

app_path = os.path.dirname(os.path.abspath('__file__'))

def vis_3D(filename):
	'''
	this function visualizes
	RMSE, alpha, window_size
	'''
	df = pd.read_csv(filename)

	a = df['alpha'].tolist()
	b = df['window_size'].tolist()
	c = df['rmse'].tolist()

	a = np.asarray(a)
	b = np.asarray(b)
	c = np.asarray(c)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')


	ax.scatter(a, b, c)

	ax.set_xlabel('Alpha')
	ax.set_ylabel('Window Per')
	ax.set_zlabel('RMSE')

	plt.show()

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	# # Plot the surface.
	# surf = ax.plot_surface(np.asarray(a), np.asarray(b), np.asarray(c), cmap=cm.coolwarm,
	#                        linewidth=0, antialiased=False)


	# ax.set_zlim(3.0, 7.0)
	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# fig.colorbar(surf, shrink=0.5, aspect=5)

	# plt.show()

def improved_predication(original_file, input_file, output_file):
	'''
	this function generates improved predicion file
	based on original_file (orginal prediction) and input_file (delta prediction file)
	results are outputed into output_file
	'''
	# TODO, need to have orginal predictions in inputfile
	# hint: use id
	df_original = pd.read_csv(original_file)
	df_input = pd.read_csv(input_file)
	df_output = pd.DataFrame({})

def convert_str_into_time(input_list):
	result_list = []
	for i in input_list:
		result_list.append(datetime.strptime(i, '%Y--%m--%d'))
	return result_list

def vis_error_prediction_PI(input_file, fig_title):
	'''
	this function vis error prediction boundary
	'''
	df = pd.read_csv(input_file)
	x_id = convert_str_into_time(df['time'].tolist())
	lower = df['lower'].tolist()
	upper = df['upper'].tolist()
	prediction = df['prediction'].tolist()
	ground_truth = df['ground_truth'].tolist()

	fig, ax = plt.subplots()
	ax.plot(x_id,lower, '-',linewidth=2, label='lower_bound')
	ax.plot(x_id,upper, '--',linewidth=2, label='upper_bound')
	ax.plot(x_id,prediction, ':',linewidth=2, label='predict_error')
	ax.plot(x_id,ground_truth, 'r--',linewidth=2, label='ground_truth')
	legend = ax.legend(loc='lower right', shadow=True)
	# legend = ax.legend(bbox_to_anchor=(0., 0.0, 1.0, .050), loc=3, ncol=1, mode="expand", borderaxespad=0.)

	plt.xlabel('time')
	plt.ylabel('value')
	plt.title(fig_title)
	plt.show()

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


def vis_improved_prediction_PI(original_model_output, input_file, fig_title, output_file = 'improved_predict_vs_obs.csv'):
	'''
	this function vis improved prediction boundary
	'''
	df_origin = pd.read_csv(original_model_output)
	df = pd.read_csv(input_file)
	time_list = df['time'].tolist()

	truth,origin_pred = collect_corresponding_obs_pred(df_origin,time_list)
	x_id = convert_str_into_time(time_list)
	lower_error = df['lower'].tolist()
	lower = [x + y for x, y in zip(lower_error,origin_pred)]

	upper_error = df['upper'].tolist()
	upper = [x + y for x, y in zip(upper_error,origin_pred)]

	prediction_error = df['prediction'].tolist()
	prediction = [x + y for x, y in zip(prediction_error,origin_pred)]

	ground_truth = truth

	fp = open(output_file,'w')
	fp.write('improved_pred,obs'+'\n')
	for i in range(len(prediction)):
		fp.write(str(prediction[i])+','+str(truth[i])+'\n')
	fp.close()

	fig, ax = plt.subplots()
	ax.plot(x_id,lower, '-',linewidth=2, label='lower_bound')
	ax.plot(x_id,upper, '--',linewidth=2, label='upper_bound')
	ax.plot(x_id,prediction, ':',linewidth=2, label='improved_prediction')
	ax.plot(x_id,truth, 'r--',linewidth=2, label='ground_truth')
	legend = ax.legend(loc='upper right', shadow=True)
	# legend = ax.legend(bbox_to_anchor=(0., 0.0, 1.0, .050), loc=3, ncol=1, mode="expand", borderaxespad=0.)

	plt.xlabel('time')
	plt.ylabel('value')
	plt.title(fig_title)
	plt.show()
	# fig.savefig(app_path+'/'+fig_title+'.png')

def vis_window_vs_rmse(win_list, origin_rmse, improved_rmse, fig_title='Original vs Improved RMSE'):
	'''
	this vis compare origin and improved rmse
	'''
	fig, ax = plt.subplots()
	ax.plot(win_list,origin_rmse, '-',linewidth=2, label='origin_rmse')
	ax.plot(win_list,improved_rmse, '--',linewidth=2, label='GB_tree_improved_rmse')

	legend = ax.legend(loc='lower right', shadow=True)
	# legend = ax.legend(bbox_to_anchor=(0., 0.0, 1.0, .050), loc=3, ncol=1, mode="expand", borderaxespad=0.)

	plt.xlabel('window size')
	plt.ylabel('rmse')
	plt.title(fig_title)
	plt.show()

def vis_original_truth_pred(original_file, smooth_file ,obs_name='runoff_obs',pred_name='basin_cfs_pred',fig_title='smooth original prediction'):
	'''
	'''
	df_origin = pd.read_csv(original_file)
	df = pd.read_csv(smooth_file)
	obs = df[obs_name].tolist()
	pred = df[pred_name].tolist()
	origin_pred = df_origin[pred_name].tolist()

	# hardcoded timestamp list
	year = df['year'].tolist()
	month = df['month'].tolist()
	day = df['day'].tolist()
	time = []
	for i in range(len(year)):
		time.append(datetime.strptime(str(year[i])+'-'+str(month[i])+'-'+str(day[i]), '%Y-%m-%d'))
	

	fig, ax = plt.subplots()
	ax.plot(time,obs, ':',linewidth=2, label='truth')
	ax.plot(time,pred, '-',linewidth=2, label='smooth_pred')
	ax.plot(time,origin_pred, '--',linewidth=2, label='original_pred')

	legend = ax.legend(loc='upper right', shadow=True)
	# legend = ax.legend(bbox_to_anchor=(0., 0.0, 1.0, .050), loc=3, ncol=1, mode="expand", borderaxespad=0.)

	plt.xlabel('time')
	plt.ylabel('value')
	plt.title(fig_title)
	plt.show()

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



def vis_bound_file(original_model_output, input_file, fig_title, output_file = 'improved_predict_vs_obs.csv'):
	'''
	if bound file has everything then use this one
	'''
	df_origin = pd.read_csv(original_model_output)
	df = pd.read_csv(input_file)
	time_list = df['time'].tolist()

	truth,origin_pred = collect_corresponding_obs_pred(df_origin,time_list)
	x_id = convert_str_into_time(time_list)
	lower = df['lower'].tolist()

	upper = df['upper'].tolist()

	prediction = df['prediction'].tolist()

	ground_truth = truth

	fp = open(output_file,'w')
	fp.write('improved_pred,obs'+'\n')
	for i in range(len(prediction)):
		fp.write(str(prediction[i])+','+str(truth[i])+'\n')
	fp.close()

	fig, ax = plt.subplots()
	ax.plot(x_id,lower, '-',linewidth=2, label='lower_bound')
	ax.plot(x_id,upper, '--',linewidth=2, label='upper_bound')
	ax.plot(x_id,prediction, ':',linewidth=2, label='improved_prediction')
	ax.plot(x_id,truth, 'r--',linewidth=2, label='ground_truth')
	ax.plot(x_id,origin_pred, '-',linewidth=2, label='original_predication')
	legend = ax.legend(loc='upper right', shadow=True)
	# legend = ax.legend(bbox_to_anchor=(0., 0.0, 1.0, .050), loc=3, ncol=1, mode="expand", borderaxespad=0.)

	plt.xlabel('time')
	plt.ylabel('value')
	plt.title(fig_title)
	plt.show()

def vis_measurement_metrix_bar_graph(original_file, result_file):
	'''
	bar graph [improved_rmse,pbias,cd,nse,original_rmse,pbias,cd,nse]
	'''
	df_origin = pd.read_csv(original_file)
	df = pd.read_csv(result_file)
	time_list = df['time'].tolist()

	truth,origin_pred = collect_corresponding_obs_pred(df_origin,time_list)
	prediction = df['prediction'].tolist()
	if len(prediction)!=len(origin_pred):
		print "Error! Len does not match"
		return
	else:
		improved_rmse = get_root_mean_squared_error(prediction,truth)
		improved_pbias = get_pbias(prediction,truth)
		improved_cd = get_coeficient_determination(prediction,truth)
		improved_nse = get_nse(prediction,truth)
		original_rmse = get_root_mean_squared_error(origin_pred,truth)
		original_pbias = get_pbias(origin_pred,truth)
		original_cd = get_coeficient_determination(origin_pred,truth)
		original_nse = get_nse(origin_pred,truth)

		n_groups = 4

		improved = (improved_rmse, improved_pbias, improved_cd, improved_nse)

		original = (original_rmse, original_pbias, original_cd, original_nse)

		fig, ax = plt.subplots()

		index = np.arange(n_groups)
		bar_width = 0.35

		opacity = 0.4
		error_config = {'ecolor': '0.3'}

		rects1 = plt.bar(index, improved, bar_width,
						 alpha=opacity,
						 color='b',
						 error_kw=error_config,
						 label='Improved')

		rects2 = plt.bar(index + bar_width, original, bar_width,
						 alpha=opacity,
						 color='r',
						 error_kw=error_config,
						 label='Original')

		plt.xlabel('Measurements')
		plt.ylabel('Values')
		plt.title('measurements')
		plt.xticks(index + bar_width / 2, ('RMSE', 'PBIAS', 'CD', 'NSE'))
		plt.legend()

		plt.tight_layout()
		plt.show()

# vis_error_prediction_PI('bound.csv','predicted_error_PI')
# vis_improved_prediction_PI('prms_input.csv', 'bound.csv','predicted_error_PI','improved_predict_vs_obs.csv')
# vis_improved_prediction_PI('prms_input.csv', '/home/host0/Downloads/03_08_boxcox_bound.csv','0.3 alpha 0.8 window size boxcox_PI','03_08_improved_predict_vs_obs.csv')

# improved_predication('prms_input.csv','bound.csv','improved_PI')
# vis_3D('rmse_all_results.csv')

# vis_improved_prediction_PI('prms_input.csv', '/home/host0/Downloads/05_bound.csv','0.6 alpha, 0.5 window size GB tree logsinh_PI transform','05_logsinh_improved_predict_vs_obs.csv')

# win_list = [0.4,0.5,0.6,0.7,0.8,0.9]
# origin_rmse = [5.28813754801,5.77548835599,6.08245036506,6.54099071958,7.42383352014,2.16472090388]
# improved_rmse = [3.37988118301,2.46425126765,4.5716992025,5.15222647189,5.22045868218,2.89334929188]
# vis_window_vs_rmse(win_list, origin_rmse, improved_rmse)


# smooth_origin_input_cse('data/prms_input.csv', 'data/smoothed_prms_input.csv', 10)
# vis_original_truth_pred('data/prms_input.csv', 'data/smoothed_prms_input.csv', fig_title='smooth original prediction threshold 10')

# smooth_origin_input_cse('prms_input.csv', 'smoothed_prms_input.csv', 10)

# vis_original_truth_pred('prms_input.csv', 'smoothed_prms_input.csv')
# vis_bound_file('data/tmp_cali.csv', 'sub_results/bound.csv','0.1 alpha, 0.5 window size GB tree logsinh_PI transform','05_logsinh_improved_predict_vs_obs.csv')



vis_measurement_metrix_bar_graph('data/prms_input.csv', '/home/host0/Desktop/05_06_threshold_20_sub_results/bound.csv')