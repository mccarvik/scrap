// Load mongoose package
var mongoose = require('mongoose')
// Connect to MongoDB and create/use database called todoAppTest
mongoose.connect('mongodb://localhost/todoAppTest');

// todo schema
var TodoSchema = new mongoose.Schema({
  name: String,
  completed: Boolean,
  note: String,
  updated_at: { type: Date, default: Date.now },
});

// Get an instance
// Need to put in the name and schema, kind of annoying
var Todo = mongoose.model('Todo', TodoSchema);

// callback function to avoid duplicating it all over
var callback = function (err, data) {
  if (err) { return console.error(err); }
  else { console.log(data); }
};

// Model.remove(conditions, update, [options], [callback])
// There is nothing returned by these functions so no callback used

// Remove by given ID
Todo.findByIdAndRemove('5934507bea6cb77a0a8ef379');
// remove multiple tasks from complete false to true
Todo.remove({ name: /master/i }, { completed: true }, { multi: true });
//Model.findOneAndRemove([conditions], [update], [options], [callback])
Todo.findOneAndRemove({name: /JS$/ }, {completed: false});
