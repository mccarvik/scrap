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

// *** Read and Query
// find all
Todo.find(function (err, todos) {
    if (err) return console.error(err);
        console.log(todos)
});

// callback function to avoid duplicating it all over
var callback = function (err, data) {
  if (err) { return console.error(err); }
  else { console.log(data); }
}

// // Get ONLY completed tasks
Todo.find({completed: true }, callback);
// // Get all tasks ending with `JS`
Todo.find({name: /JS$/ }, callback);


var oneYearAgo = new Date();
oneYearAgo.setYear(oneYearAgo.getFullYear() - 1);
// Get all tasks staring with `Master`, completed
Todo.find({name: /^Master/, completed: true }, callback);
// Get all tasks staring with `Master`, not completed and created from year ago to now...
Todo.find({name: /^Master/, completed: false }).where('updated_at').gt(oneYearAgo).exec(callback);
