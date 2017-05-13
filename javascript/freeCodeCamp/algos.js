
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

function findLongestWord(str) {
  var arr = str.split(" ");
  var longest = "";
  for (var i=0; i < arr.length; i++) {
    var string = arr[i];
    if (string.length > longest.length) {
      longest = string;
    }
  }
  return longest.length;
}
// console.log(findLongestWord("The quick brown fox jumped over the lazy dog"));

function titleCase(str) {
  var arr = str.split(" ");
  var ret = [];
  for (var a=0; a < arr.length; a++) {
    var spl = arr[a].split("");
    spl[0] = spl[0].toUpperCase();
    for (var b=1; b < spl.length; b++) {
      spl[b] = spl[b].toLowerCase();
    }
    ret.push(spl.join(""));
  }
  return ret.join(" ");
}
console.log(titleCase("I'm a little tea pot"));

function largestOfFour(arr) {
  var ret = [];
  for (var i=0; i < arr.length; i++) {
    var max = 0;
    for (var k=0; k <arr[i].length; k++) {
      if (arr[i][k] > max) {
        max = arr[i][k];
      }
    }
    ret.push(max);
  }
  return ret;
}

// console.log(largestOfFour([[4, 5, 1, 3], [13, 27, 18, 26], [32, 35, 37, 39], [1000, 1001, 857, 1]]);)

