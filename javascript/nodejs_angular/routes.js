/********
 * Routes
 ********/

exports.index = function(req, res){
    console.log('Got to Index');
    res.render('index');
};
 
exports.partials = function (req, res) {
  var name = req.params.name;
  res.render('partials/' + name);
};