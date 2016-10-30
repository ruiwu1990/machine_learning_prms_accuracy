$(document).ready(function(){

	$('#get_error_result_id').on('click',function(){
		  $.ajax({
		    type: 'GET',
		    url: '/api/decision_tree_data',
		    success: function (data) {
		      var received_json = JSON.parse(data);
		       $('#result_info_id').html('<div style="margin-left: 60px;">The original rmse is: <span style="color:blue;font-weight: bold;">'+received_json['original_rmse']+'</span><br>'+
		      							'The improved rmse is: <span style="color:blue;font-weight: bold;">'+received_json['improved_rmse']+'</span><br>'+
		      							'The delta error rmse is : <span style="color:blue;font-weight: bold;">'+received_json['delta_error_rmse']+'</span><br>'+
		      							'The original pbias is : <span style="color:blue;font-weight: bold;">'+received_json['original_pbias']+'</span><br>'+
		      							'The improved pbias is : <span style="color:blue;font-weight: bold;">'+received_json['improved_pbias']+'</span><br>'+
		      							'The delta error pbias is: <span style="color:blue;font-weight: bold;">'+received_json['delta_error_pbias']+'</span><br>'+
		      							'The original cd is: <span style="color:blue;font-weight: bold;">'+received_json['original_cd']+'</span><br>'+
		      							'The improved cd is: <span style="color:blue;font-weight: bold;">'+received_json['improved_cd']+'</span><br>'+
		      							'The delta error cd is: <span style="color:blue;font-weight: bold;">'+received_json['delta_error_cd']+'</span><br>'+
		      							'The original nse is: <span style="color:blue;font-weight: bold;">'+received_json['original_nse']+'</span><br>'+
		      							'The improved nse is: <span style="color:blue;font-weight: bold;">'+received_json['improved_nse']+'</span><br>'+
		      							'The delta error nse is: <span style="color:blue;font-weight: bold;">'+received_json['delta_error_nse']+'</span><br></div><br>'); 
		     
		      original_p_list = received_json['original_p_list'];
		      improved_p_list = received_json['improved_p_list'];
		      o_list = received_json['o_list'];
		      draw_line_chart(original_p_list,improved_p_list,o_list);
		    }
		  });
	});
	// these functions should be moved into decision tree vis js
	function draw_line_chart(o_p_list,i_p_list,o_list){
		// prepare dataset
		var data_array = [['index','original_p_list','improved_p_list','o_list']];
		for(var i=0; i< o_p_list.length; i++){
			data_array.push([i,o_p_list[i],i_p_list[i],o_list[i]]);	
		}
		

		google.charts.load('current', {'packages':['corechart']});
		google.charts.setOnLoadCallback(drawChart);

		function drawChart() {
	        var data = google.visualization.arrayToDataTable(data_array);
			var options = {
	          title: 'Predication VS Obervation',
	          curveType: 'function',
	          legend: { position: 'bottom' }
	        };

	        var chart = new google.visualization.LineChart(document.getElementById('line_chart_id'));

	        chart.draw(data, options);
	    }

	}
	


});

