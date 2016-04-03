// Width and Height
var margin = {top: 20, right: 20, bottom: 30, left: 40},
width = 600 - margin.left - margin.right,
height = 300 - margin.top - margin.bottom,
maxR = 15;


// Setup x
var xValue = function(d){return d.Prostitution},
xScale = d3.scale.linear().range([0,width]),
xMap = function(d){return xScale(xValue(d))},
xAxis = d3.svg.axis().scale(xScale).orient('bottom').ticks(6);

// setup y
var yValue = function(d){return d.Theft},
yScale = d3.scale.linear().range([height,0]),
yMap = function(d){return yScale(yValue(d))},
yAxis = d3.svg.axis().scale(yScale).orient('left').ticks(5);


// setup r
var rValue = function(d){return d.Total},
rScale = d3.scale.linear().range([3,maxR]),
rMap = function(d){return rScale(rValue(d))};



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
  .attr('fill','Blue');
};

// Load datasets
d3.csv('csv/SFPDmodified2003.csv',function(error,data){
  if (error) {
    console.log(error)
  }
  data.forEach(function(d){
    d.Prostitution = +d.Prostitution
    d.Theft = +d.Theft
    d.Total = +d.Total
  })

  xScale.domain([0, d3.max(data, xValue)+1]);
  yScale.domain([0, d3.max(data, yValue)+1]);
  rScale.domain([d3.min(data,rValue),d3.max(data,rValue)])

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
  .attr('fill','Blue');

})


d3.select("#chart2").selectAll('Button')
.on('click',function(d){

  var paragraphID = d3.select(this).attr("id");

  if (paragraphID=='2015'){
    var datafile = 'csv/SFPDmodified2015.csv'
  } else if (paragraphID=='2003') {
    var datafile = 'csv/SFPDmodified2003.csv'
  }

  d3.csv(datafile,function(error,data){
    if (error) {
      console.log(error)
    }
    data.forEach(function(d){
      d.Prostitution = +d.Prostitution
      d.Theft = +d.Theft
      d.Total = +d.Total
    })

    updateChart2(data)


  });

});
