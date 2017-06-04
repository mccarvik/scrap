// Load mongoose package
var mongoose = require('mongoose');
// Connect to MongoDB and create/use database called todoAppTest
mongoose.connect('mongodb://localhost/todoAppTest');
// Create a schema
var TodoSchema = new mongoose.Schema({
  name: String,
  completed: Boolean,
  note: String,
  updated_at: { type: Date, default: Date.now },
});
// Create a model based on the schema
var Todo = mongoose.model('Todo', TodoSchema);


// Create a todo in memory
var todo = new Todo({name: 'adad', completed: false, note: '3'});
// Save it to database
console.log('h');
todo.save(function(err){
  if(err) {
    // console.log(err);
    console.log('e');
  } else {
    // console.log(todo);
    console.log('s');
  }
});

// *** Can create and save an entry in one line like this
Todo.create({name: 'Master', completed: true, note: 'this is one'}, function(err, todo){
    if(err) console.log(err);
    else console.log(todo);
});

