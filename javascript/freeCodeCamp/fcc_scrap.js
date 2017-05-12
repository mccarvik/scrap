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
// console.log(caseInSwitch(1));

// Objects
var testObj = {
  "hat" : "ballcap",
  "shirt" : "jersey",
  "shoes" : "cleats",
  'formats' : [
      'short sleeve',
      'long sleeve'
    ],
  'other': {
        'a': 'nested',
        'b' : {
          'c' : "even more nested"
        },
        'c' : 'still nested'
  }
};
// console.log(testObj.hat);
// console.log(testObj['shirt']);
testObj['shoes'] = 'spikes';
var c = 'shoes';
// console.log(testObj[c]);
delete testObj['hat'];
if (testObj.hasOwnProperty('formats')) {
  // console.log(testObj['formats']);
} else {
  // console.log('Not Found');
}
// nested objects typically use dot notation until final level
// console.log(testObj.other.b['c']);

// Loops
var myArray = [];
for (var i=1; i<6; i++){
  myArray.push(i);
}
// console.log(myArray)
myArray = [];
var i = 0;
while (i<5) {
  myArray.push(i);
  i++;
}
// console.log(myArray)

// Exercise: Profile Lookup
//Setup
var contacts = [
    {
        "firstName": "Akira",
        "lastName": "Laine",
        "number": "0543236543",
        "likes": ["Pizza", "Coding", "Brownie Points"]
    },
    {
        "firstName": "Harry",
        "lastName": "Potter",
        "number": "0994372684",
        "likes": ["Hogwarts", "Magic", "Hagrid"]
    },
    {
        "firstName": "Sherlock",
        "lastName": "Holmes",
        "number": "0487345643",
        "likes": ["Intriguing Cases", "Violin"]
    },
    {
        "firstName": "Kristian",
        "lastName": "Vos",
        "number": "unknown",
        "likes": ["Javascript", "Gaming", "Foxes"]
    }
];


function lookUpProfile(firstName, prop) {
// Only change code below this line
  for (var i=0; i<contacts.length; i++) {
    if (contacts[i].firstName === firstName) {
      if (contacts[i].hasOwnProperty(prop)) {
        return contacts[i][prop];
      } else {
        return "No such property";
      } 
    }
  }
  return "No such contact"; 
// Only change code above this line
};

// Change these values to test your function
// console.log(lookUpProfile("Kristian", "lastName"));

// Random
var a = Math.floor(Math.random()*10);

// Regular Expressions (Regex)
// g --> global, return all matches not just the first, i--> ignore case, 
// \d --> find digits, \s --> find white space, + --> match one or more digits
// Capital S or D, inverts the expression
var testString = "Ada Lovelace and Charles Babbage designed the first computer and the software that would have run on it.";
var expression = /and/gi;
var andCount = testString.match(expression);
console.log(andCount)

testString = "There are 3 cats but 4 dogs.";
expression = /\d+/g;
console.log(testString.match(expression));

testString = "How many spaces are there in this sentence?";
expression = /\s+/g;
console.log(testString.match(expression).length);

testString = "How many non-space characters are there in this sentence?";
expression = /\S/g;  // Change this line
console.log(testString.match(expression));


