var express=require('express');
var app=express();
// var server=app.listen(8000, '0.0.0.0');
app.listen(8080, '0.0.0.0');

app.route('/Node').get(function(req, res)
{
    res.send("Tutorial on Node");
});
app.route('/Angular').get(function(req, res)
{
    res.send("Tutorial on Angular");
});
app.get('/',function(req, res)
{
    res.send('Welcome to Guru99 Tutorials');
});

console.log('Server running at http://0.0.0.0:8080/');