
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

function diffArray(arr1, arr2) {
  var newArr = [];
  both = [];
  for (var i=0; i < arr1.length; i++) {
    for (var k=0; k < arr2.length; k++) {
      if (arr1[i] === arr2[k]) {
        both.push(arr1[i]);
        break;
      }
    }
  }
  for (var b=0; b < both.length; b++) {
    arr1.splice(arr1.indexOf(both[b]), 1);
    arr2.splice(arr2.indexOf(both[b]), 1);
  }
  return arr1.concat(arr2);
}
// console.log(diffArray([1, 2, 3, 5], [1, 2, 3, 4, 5]));

