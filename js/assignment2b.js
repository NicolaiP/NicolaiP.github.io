// Width and Height
var margin = {top: 20, right: 20, bottom: 30, left: 40},
width = 600 - margin.left - margin.right,
height = 300 - margin.top - margin.bottom,
maxR = 15;


// Setup x
var xValue = function(d){return d.Prostitution2003},
xScale = d3.scale.linear().range([0,width]),
xMap = function(d){return xScale(xValue(d))},
xAxis = d3.svg.axis().scale(xScale).orient('bottom').ticks(6);

// setup y
var yValue = function(d){return d.Theft2003},
yScale = d3.scale.linear().range([height,0]),
yMap = function(d){return yScale(yValue(d))},
yAxis = d3.svg.axis().scale(yScale).orient('left').ticks(5);


// setup r
var rValue = function(d){return d.Total2003},
    rMap = function(d){return rValue(d)/1200} // 1200 is prop


// Create SVG chart-area
var chart2 = d3.select("#chart2")
.append("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", "translate(" + margin.left + "," + margin.top + ")");


// Update function
function updateChart2(data){
  chart2.selectAll('circle')
  .data(data)
  .transition()
  .duration(1000)
  .attr("r", rMap)
  .attr("cx", xMap)
  .attr("cy", yMap)
  .attr('fill','Blue')
  .append("title")
  .text(function(d){
    return d.District
  });
};

// Load datasets
d3.csv('csv/SFPDmodified.csv',function(error,data){
  if (error) {
    console.log(error)
  }
  data.forEach(function(d){
    d.Prostitution2003 = +d.Prostitution2003
    d.Theft2003 = +d.Theft2003
    d.Total2003 = +d.Total2003
    d.Prostitution2015 = +d.Prostitution2015
    d.Theft2015 = +d.Theft2015
    d.Total2015 = +d.Total2015
  })

  var tempX1 = d3.max(data,function(d){return d.Prostitution2003}),
      tempX2 = d3.max(data,function(d){return d.Prostitution2015}),
      tempY1 = d3.max(data,function(d){return d.Theft2003}),
      tempY2 = d3.max(data,function(d){return d.Theft2015});

  xScale.domain([0, d3.max([tempX1,tempX2])+1]);
  yScale.domain([0, d3.max([tempY1,tempY2])+1]);

  // x-axis
  chart2.append("g")
  .attr("class", "x axis")
  .attr("transform", "translate(0," + height + ")")
  .call(xAxis)
  .append("text")
  .attr("class", "label")
  .attr("x", width)
  .attr("y", -6)
  .style("text-anchor", "end")
  .text("Prostitution");

  // y-axis
  chart2.append("g")
  .attr("class", "y axis")
  .call(yAxis)
  .append("text")
  .attr("class", "label")
  .attr("transform", "rotate(-90)")
  .attr("y", 6)
  .attr("dy", ".71em")
  .style("text-anchor", "end")
  .text("Vehicle Theft");

  // draw stuff
  chart2.append("g")
  .attr('id','circles')
  .selectAll('circles')
  .data(data)
  .enter().append("circle")
  .attr("r", rMap)
  .attr("cx", xMap)
  .attr("cy", yMap)
  .attr('fill','Blue')
  .append("title")
  .text(function(d){
    return d.District
  });


d3.selectAll('Button')
  .on('click',function(d){

    var paragraphID = d3.select(this).attr("id");

    if (paragraphID=='2015'){
      xValue = function(d){return d.Prostitution2015}
      yValue = function(d){return d.Theft2015}
      rValue = function(d){return d.Total2015}

    } else if (paragraphID=='2003') {
      xValue = function(d){return d.Prostitution2003}
      yValue = function(d){return d.Theft2003}
      rValue = function(d){return d.Total2003}
    }

      updateChart2(data)

  });
});
