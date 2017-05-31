var express=require('express');
var app=express();
app.get('/',function(req,res)
{
    res.send('Hello World!');
});
// var server=app.listen(8000, '0.0.0.0');
app.listen(8080, '0.0.0.0');

console.log('Server running at http://0.0.0.0:8080/');