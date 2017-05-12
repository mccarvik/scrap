// Constructor Functions
// create objects using constructor functions
// capitalized name to make it clear that it is a constructor
var Car = function(w, s, e) {
  this.wheels = w;
  this.seats = s;
  this.engines = e;
  
  // this is a private variable
  var speed = 10;

  // these are public methods
  this.accelerate = function(change) {
    speed += change;
  };
  this.decelerate = function() {
    speed -= 5;
  };
  this.getSpeed = function() {
    return speed;
  };
};
// important to use 'new' keyword when calling constructor
// how Javascript knows to create new object
// all the references to this inside constructor should refer to this new object
var myCar = new Car(4, 1, 1);
myCar.nickname= "dog";
// console.log(myCar);

// callback function --> a function passed to another function as a parameter
// and the callback function is called (or executed) inside the otherFunction
var oldArray = [1,2,3,4,5];
// "Add 3" is the callback function
console.log(oldArray.map(function(val) { return val + 3; }));
console.log(oldArray.reduce(function(prev, cur) { return prev+cur; }));
console.log(oldArray.filter(function(val) { return val < 3; }));
// Sort function
var array = [1, 12, 21, 2];
console.log(array.sort(function(a,b) { return b - a;}));
console.log(array.reverse());

var oldArray = [1,2,3];
var concatMe = [4,5,6];
console.log(oldArray.concat(concatMe));

var string = "Split me into an array";
console.log(string.split(" "));
var joinMe = ["Split","me","into","an","array"];
console.log(joinMe.join(" "));