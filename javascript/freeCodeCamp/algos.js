
function reverseString(str) {
  var rev = [];
  for (var i=0; i < str.length; i++) {
    rev.unshift(str[i]);
  }
  return rev.join("")
}
console.log(reverseString("hello"));

