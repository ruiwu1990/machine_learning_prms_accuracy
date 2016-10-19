import numpy as np
import netCDF4

def get_run_off(input_file):
	'''
	this function gets the run_off (observed data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	observed_data = list(prms_fp.variables['runoff_1'][:])
	prms_fp.close()
	return observed_data

def get_basin_cfs(input_file):
	'''
	this function gets the basin_cfs (predicted data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	predicted_data = list(prms_fp.variables['basin_cfs_1'][:])
	prms_fp.close()
	return predicted_data

def get_time(input_file):
	'''
	this function gets the basin_cfs (predicted data) from the
	stat.nc file and return it in a list
	'''
	prms_fp = netCDF4.Dataset(input_file,'r')
	time_data = list(prms_fp.variables['time'][:])
	prms_fp.close()
	return time_data

def output_csv(input_file):
	'''
	output time, observed data, and predicted data, delta error (observed - predicted)
	into a csv file
	'''
	fp = open('data.csv','w')
	time_list = get_time(input_file)
	predicted_list = get_basin_cfs(input_file)
	observed_list = get_run_off(input_file)
	list_len = len(time_list)
	for count in range(list_len):
		fp.write(','.join([str(time_list[count]),str(predicted_list[count]),str(observed_list[count]),str(observed_list[count]-predicted_list[count])])+'\n')
	fp.close()

input_file = 'stat.nc'
output_csv(input_file)