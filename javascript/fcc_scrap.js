// To run type "node" in command line and then javascript file name

function reusuableFunction() {
    console.log("Hi World");
}
// reusuableFunction();

// Variables which are used without the var keyword are automatically created 
// in the global scope. You should always declare your variables with var.
var myGlobal = 10;

function fun1() {
  oopsGlobal = 5;
}

function fun2() {
  var output = "";
  if (typeof myGlobal != "undefined") {
    output += "myGlobal: " + myGlobal;
  }
  if (typeof oopsGlobal != "undefined") {
    output += " oopsGlobal: " + oopsGlobal;
  }
  console.log(output);
}
// fun1();
// fun2();

// It is possible to have both local and global variables with the same name
// When you do this, the local variable takes precedence over the global
// Dont do this
var outerWear = "T-Shirt";

function myOutfit() {
  var outerWear = "sweater"
  return outerWear;
}
// myOutfit();

// Strict equality (===) is the counterpart to the equality operator (==). 
// Unlike (==), (===) tests both the data type and value of the compared elements
// use === almost always to be safe
function strictVsRegEquality() {
  var num = 0;
  var obj = new String('0');
  var str = '0';

  console.log(num === num); // true
  console.log(obj === obj); // true
  console.log(str === str); // true
  
  console.log(num === obj); // false
  console.log(num==obj);    // true
  console.log(num === str); // false
  console.log(num == str); // true
  console.log(obj === str); // false
  console.log(obj == str); // true
  console.log(null === undefined); // false
  console.log(obj === null); // false
  console.log(obj === undefined); // false
}
// strictVsRegEquality()

// Switch statement, syntax example
function caseInSwitch(val) {
  var answer = "";
  // Only change code below this line
  switch (val) {
    case 1:
      answer = "alpha";
      break;
    case 2:
      answer = "beta";
      break;
    case 3:
      answer = "gamma";
      break;
    case 4:
    case 5:
      answer = "delta";
      break;
    default:
      answer = "omega";
      break;
  }  
  return answer;  
}
console.log(caseInSwitch(1));