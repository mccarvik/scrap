
function sumAll(arr) {
  var max = Math.max.apply(null, arr);
  var min = Math.min.apply(null, arr);
  var sum = 0;
  for (var i = min; i <= max; i++) {
    sum += i;
  }
  return sum;
}
// console.log(sumAll([1, 4]));