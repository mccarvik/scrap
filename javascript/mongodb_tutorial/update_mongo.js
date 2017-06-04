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
}

// Model.update(conditions, update, [options], [callback])
// update multiple tasks from complete false to true
Todo.update({ name: /master/i }, { completed: true }, { multi: true }, callback);
//Model.findOneAndUpdate([conditions], [update], [options], [callback])
Todo.findOneAndUpdate({name: /JS$/ }, {completed: false}, callback);