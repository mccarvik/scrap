var express = require('express'),
    rts     = require('./routes'),
    http    = require('http'),
    path    = require('path');
    
var app = module.exports = express();
app.engine('html', require('ejs').renderFile);
app.listen(8080, '0.0.0.0');
console.log('Server running at http://0.0.0.0:8080/');

app.set('port', process.env.PORT || 8080);
app.set('views', __dirname + '/views');
app.set('view engine', 'html');
// app.use(express.logger('dev'));
app.use(express.static(path.join(__dirname, 'public')));

// if (app.get('env') === 'development') {
//   app.use(express.errorHandler());
// }
app.route('/').get(rts.index);
app.route('/partials/:name').get(rts.partials);
app.route('*').get(rts.index);

