
function reverseString(str) {
  var rev = [];
  for (var i=0; i < str.length; i++) {
    rev.unshift(str[i]);
  }
  return rev.join("")
}
// console.log(reverseString("hello"));

function factorialize(num) {
  var prod=1;
  for (var i=1; i<=num; i++) {
    prod *= i;
  }
  
  return prod;
}
// console.log(factorialize(5));



function palindrome(str) {
  // format string
  var expression = /[a-zA-Z0-9]+/g;
  str = str.match(expression).join("");
  expression = /\S+/g;
  // console.log(str.match(expression));
  str = str.match(expression).join("");
  var arr = str.split("");
  var len = (Math.floor(arr.length / 2));
  for (var i=0; i < len; i++) {
    if (arr.pop().toUpperCase() == arr.shift().toUpperCase()) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}
var s = "not a palindrome";
// console.log(palindrome(s));

