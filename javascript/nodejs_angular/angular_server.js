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
// This sets the root directory of the web server --> "" = current directory
app.use(express.static(path.join(__dirname, '')));


/**************
 * Start Server
 **************/
app.listen(8080, '0.0.0.0', function () {
  console.log('Server running at http://0.0.0.0:' + app.get('port'));
});


/********
 * Routes
 ********/

// Need this to eliminate favicon.ico erro
app.route('/favicon.ico').get(function(req, res) {
    res.sendStatus(204);
}); 
app.route('/').get(rts.index);
app.route('/:name').get(rts.partials);
// app.route('*').get(rts.index);

