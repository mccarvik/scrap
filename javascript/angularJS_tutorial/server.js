#!/usr/bin/env nodejs
var http = require('http');
http.createServer(function (req, res) {
  res.writeHead(200, {'Content-Type': 'text/plain'});
  // res.write(html);
  res.end('Hello World\n');
}).listen(8080, '0.0.0.0');
// }).listen(8080, 'localhost');
// Very important to use 0.0.0.0 instead of localhost
console.log('Server running at http://localhost:8080/');