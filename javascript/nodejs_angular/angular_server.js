/*********************
 * Module dependencies
 *********************/
var express = require('express'),
    rts     = require('./routes'),
    http    = require('http'),
    path    = require('path');

/***************
 * Configuration
 ***************/
var app = module.exports = express();
app.engine('html', require('ejs').renderFile);
app.set('port', process.env.PORT || 8080);
app.set('views', __dirname + '/views');
app.set('view engine', 'html');
// app.use(express.logger('dev'));
app.use(express.static(path.join(__dirname, 'public')));


/**************
 * Start Server
 **************/
app.listen(8080, '0.0.0.0', function () {
  console.log('Server running at http://0.0.0.0:' + app.get('port'));
});


/********
 * Routes
 ********/
app.route('/').get(rts.index);
// app.route('/directives').get(rts.directives);
// app.route('/controller').get(rts.controller);
// app.route('/directives').get(rts.partials('directives'));
// app.route('/controllers').get(rts.partials('controllers'));
// app.route('/partials/:name').get(rts.partials);
app.route('/:name').get(rts.partials);
app.route('*').get(rts.index);

