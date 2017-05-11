// To run type "node" in command line and then javascript file name

function reusuableFunction() {
    console.log("Hi World");
}

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

fun1();
fun2();
// reusuableFunction();