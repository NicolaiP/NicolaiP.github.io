// Width and Height
var margin3 = {top: 20, right: 20, bottom: 100, left: 40},
width3 = 600 - margin3.left - margin3.right,
height3 = 400 - margin3.top - margin3.bottom;

// Setup x
var xValue3 = function(d){return d.District;},
xScale3 = d3.scale.ordinal().rangeRoundBands([0,width3],0.05),
xMap3 = function(d){return xScale3(xValue3(d))},
xAxis3 = d3.svg.axis().scale(xScale3).orient('bottom');

// Setup y
var yValue3 = function(d){return d.Prostitution},
yScale3 = d3.scale.linear().range([height3,0]),
yMap3 = function(d){return yScale3(yValue3(d))},
yHeight = function(d){return height3-yScale3(yValue3(d))},
yAxis3 = d3.svg.axis().scale(yScale3).orient('left');

// Key Binder
var key = function(d){
  return d.District
};

// Update functions
function updateChart3(data){
  chart3.selectAll('.bar')
  .data(data)
  .transition()
  .duration(1000)
  .attr("x", xMap3)
  .attr("y", yMap3)
  .attr("width", xScale3.rangeBand())
  .attr("height", yHeight)
  .attr('fill','steelblue');
};

// Update Axis
function updateAxis3(xAxis3,yAxis3){
  chart3.select(".y.axis")
  .transition()
  .duration(1000)
  .call(yAxis3);
};




// Create SVG chart-area
var chart3 = d3.select("#chart3")
.append("svg")
.attr("width", width3 + margin3.left + margin3.right)
.attr("height", height3 + margin3.top + margin3.bottom)
.append("g")
.attr("transform", "translate(" + margin3.left + "," + margin3.top + ")");


// Load datasets
d3.csv('csv/SFPDmodified.csv',function(error,data){
  if (error) {
    console.log(error)
  }
  data.forEach(function(d){
    d.Prostitution = +d.Prostitution2003
    d.Theft = +d.Theft2003
  })

  //xScale3.domain(d3.range(data.length));
  xScale3.domain(data.map(xValue3));
  yScale3.domain([0, d3.max(data, yValue3)+1]);

  // x-axis
  chart3.append("g")
  .attr("class", "x axis")
  .attr("transform", "translate(0," + height3 + ")")
  .call(xAxis3)
  .selectAll("text")
  .attr("y", 0)
  .attr("x", 9)
  .attr("dy", ".35em")
  .attr("transform", "rotate(90)")
  .style("text-anchor", "start");

  // y-axis
  chart3.append("g")
  .attr("class", "y axis")
  .call(yAxis3)
  .append("text")
  .attr("transform", "rotate(-90)")
  .attr("y", 6)
  .attr("dy", ".71em")
  .style("text-anchor", "end")
  .text("Crime count")

  //Create bars
  chart3.selectAll(".bar")
  .data(data, key)
  .enter()
  .append("rect")
  .attr("class","bar")
  .attr("x", xMap3)
  .attr("y", yMap3 )
  .attr("width", xScale3.rangeBand())
  .attr("height", yHeight)
  .attr("fill", 'steelblue')
  .on("mouseover", function(d) {

    //Get this bar's x/y values, then augment for the tooltip
    var xPosition = event.pageX - xScale3.rangeBand()
    var yPosition = event.pageY

    //Update the tooltip position and value
    d3.select("#tooltip")
    .style("left", xPosition + "px")
    .style("top", yPosition + "px")
    .select("#value")
    .text(yValue3(d));

    //Show the tooltip
    d3.select("#tooltip").classed("hidden", false);

  })
  .on("mouseout", function() {

    //Hide the tooltip
    d3.select("#tooltip").classed("hidden", true);

  })


  d3.select("#chart3").selectAll('Button')
  .on('click',function(d){

    var paragraphID = d3.select(this).attr("id");

    if (paragraphID == 'Theft'){
      yValue3 = function(d){return d.Theft}
      yScale3.domain([0, d3.max(data, yValue3)+1])
    } else {
      yValue3 = function(d){return d.Prostitution}
      yScale3.domain([0, d3.max(data, yValue3)+1])
    }

    updateChart3(data)
    updateAxis3(xAxis3,yAxis3)
  })


/*
  // Update function
  function updateChart2(data){
  svg.selectAll('circle')
  .data(data)
  .transition()
  .duration(1000)
  .attr("r", rMap)
  .attr("cx", xMap)
  .attr("cy", yMap)
  .attr('fill','Blue');
};
*/
/*
d3.selectAll('Button')
.on('click',function(d){

var paragraphID = d3.select(this).attr("id");

if (paragraphID=='2015'){
var datafile = 'csv/SFPDmodified2015.csv'
d3.csv(datafile,function(error,data){
if (error) {
console.log(error)
}
data.forEach(function(d){
d.Prostitution = +d.Prostitution
d.Theft = +d.Theft
d.Total = +d.Total
})
})
updateChart2(data)
}
else if (paragraphID == "2003") {
var datafile = 'csv/SFPDmodified2003.csv'
d3.csv(datafile,function(error,data){
if (error) {
console.log(error)
}
data.forEach(function(d){
d.Prostitution = +d.Prostitution
d.Theft = +d.Theft
d.Total = +d.Total
})
})
updateChart2(data)
}
else if (paragraphID == 'Theft') {
yValue3 = function(d){return d.Theft}
yScale3.domain([0, d3.max(data, yValue3)+1])
updateChart3(data)
updateAxis3(xAxis3,yAxis3)
}
else if (paragraphID == 'Prostitution') {
yValue3 = function(d){return d.Prostitution}
yScale3.domain([0, d3.max(data, yValue3)+1])
updateChart3(data)
updateAxis3(xAxis3,yAxis3)
}



});
*/
});
