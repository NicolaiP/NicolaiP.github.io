//Load in GeoJSON data
d3.json('json/sfpddistricts.json',function(data){


	var width = 600, height = 400;

	var chart4 = d3.select("#chart4").append("svg").attr("width", width).attr("height", height);

	var projection = d3.geo.mercator().scale(1).translate([0, 0]).precision(0);
	var path = d3.geo.path().projection(projection);
	var bounds = path.bounds(data);

	xScale4 = width / Math.abs(bounds[1][0] - bounds[0][0]);
	yScale4 = height / Math.abs(bounds[1][1] - bounds[0][1]);
	scale = xScale4 < yScale4 ? xScale4 : yScale4;

	var transl = [(width - scale * (bounds[1][0] + bounds[0][0])) / 2, (height - scale * (bounds[1][1] + bounds[0][1])) / 2];
	projection.scale(scale).translate(transl);

	chart4.selectAll("path").data(data.features).enter().append("path").attr("d", path).attr('data-id', function(d) {
		return d.id;
	}).attr('data-name', function(d) {
		return d.properties.name;
	}).style("fill", "#FB5B1F").style("stroke", "#ffffff");


	d3.csv('csv/km_data.csv',function(error,data){

		if (error) {
			console.log(error)
		}
		data.forEach(function(d){
			d.lon = +d.lon
			d.lat = +d.lat
			d.label = +d.labelK2
		})


		// possible colors
		var colors = ['blue','black','yellow','green','red','pink']

		// function for accessing colors
		function updateColors(d,i){
				if (d.label == 0) {
					return colors[0]
				}
				else if(d.label == 1) {
					return colors[1]
				}
				else if(d.label == 2) {
					return colors[2]
				}
				else if(d.label == 3) {
					return colors[3]
				}
				else if(d.label == 4) {
					return colors[4]
				}
				else if(d.label == 5) {
					return colors[5]
				}
		}

		// function for updating chart
		function updateChart4(data){
			chart4.selectAll(".dots")
			.data(data)
			.attr("fill",updateColors)
		}

		chart4.selectAll(".dots")
		.data(data)
		.enter()
		.append("circle")
		.attr("class","dots")
		.attr("cx", function(d) {
			return projection([d.lon, d.lat])[0];
		})
		.attr("cy", function(d) {
			return projection([d.lon, d.lat])[1];
		})
		.attr("r", 2)
		.attr("fill",'blue')
		.style("opacity", 0.75);


		d3.select('#chart4').selectAll('button')
			.on('click',function(d){

				var paragraphID = d3.select(this).attr("id");

				data.forEach(function(d){
				if (paragraphID == 'K2' ) {
					d.label = d.labelK2
				}
				else  if (paragraphID == 'K3' ) {
					d.label = d.labelK3
				}
				else if (paragraphID == 'K4' ) {
					d.label = d.labelK4
				}
				else if (paragraphID == 'K5' ) {
					d.label = d.labelK5
				}
				else if (paragraphID == 'K6' ) {
					d.label = d.labelK6
				}
				})
				updateChart4(data)

			})
		})

	});
