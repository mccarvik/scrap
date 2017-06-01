/********
 * Routes
 ********/

exports.index = function(req, res){
    console.log('Index');
    res.render('index');
};
 
exports.partials = function (req, res) {
  var name = req.params.name;
  console.log(name)
  res.render(name);
};