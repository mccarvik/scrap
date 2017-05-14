
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
// console.log(titleCase("I'm a little tea pot"));

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

function confirmEnding(str, target) {
  slice = str.slice(str.length - target.length, str.length);
  return (slice == target);
}
// console.log(confirmEnding("Open sesame", "same"));

function repeatStringNumTimes(str, num) {
  var ret = "";
  if (num < 1) {
    return ret;
  } else {
    for (var i=0; i < num; i++) {
      ret += str;
    }
  }
  return ret;
}
// console.log(repeatStringNumTimes("abc", 3));

function chunkArrayInGroups(arr, size) {
  var ret = [];
  while (arr.length > 0) {
    var t_arr = [];
    for (var i=0; i<size; i++){
      if (arr.length === 0) { break; }
      t_arr.push(arr.shift());
    }
    ret.push(t_arr);
  }
  return ret;
}
// console.log(chunkArrayInGroups(["a", "b", "c", "d"], 2));

function slasher(arr, howMany) {
  return arr.slice(howMany,arr.length);
}
// console.log(slasher([1, 2, 3], 2));

function mutation(arr) {
  var spl = arr[0].toUpperCase();
  var check = arr[1].split("");
  for (var i=0; i < check.length; i++){
    if (spl.indexOf(check[i].toUpperCase()) < 0) { return false; }
  }
  return true;
}
// console.log(mutation(["zyxwvutsrqponmlkjihgfedcba", "qrstu"]));


function bouncer(arr) {
  return arr.filter(function(s) {
    if(s) {
      return true;
    } else {
      return false;
    }
  });
};
// console.log(bouncer([7, "ate", "", false, 9]));

function destroyer(arr) {
  ret = arguments[0];
  for (var i=1; i<arguments.length; i++) {
    while (ret.indexOf(arguments[i]) >= 0) {
      ret.splice(ret.indexOf(arguments[i]), 1);
    }
  }
  return ret;
}
// console.log(destroyer([1, 2, 3, 1, 2, 3], 2, 3));

function getIndexToIns(arr, num) {
  arr.push(num);
  arr = arr.sort(function(a,b) { return a-b; });
  return arr.indexOf(num);
}
// console.log(getIndexToIns([3, 10, 5], 50));

function rot13(str) { // LBH QVQ VG!
  var ret = [];
  var spl = str.split("");
  for (var i=0; i < spl.length; i++) {
    if (isLetter(spl[i])) {
      if (spl[i].charCodeAt(0) < 78) {
        ret.push(String.fromCharCode(spl[i].charCodeAt(0) + 13));
      } else {
        ret.push(String.fromCharCode(spl[i].charCodeAt(0) - 13));
      }
    } else {
      ret.push(spl[i]);
    }
  }
  return ret.join("");
}

function isLetter(str) {
  return str.match(/[a-z]/i);
}
console.log(rot13("SERR PBQR PNZC"));