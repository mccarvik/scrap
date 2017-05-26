
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

function fearNotLetter(str) {
  var first = str.charCodeAt(0);
  for (var i=1; i < str.length; i++) {
    if (str.charCodeAt(i) != first + i) {
      return String.fromCharCode(first+i);
    }
  }
  return undefined;
}
// console.log(fearNotLetter("abcdefghjklmno"));

function pairElement(str) {
  var arr = str.split("");
  var ret = [];
  for (var i=0; i<arr.length; i++) {
    var temp = "";
    switch (arr[i]) {
      case "G":
        temp = "C";
        break;
      case "C":
        temp = "G";
        break;
      case "A":
        temp = "T";
        break;
      case "T":
        temp = "A";
        break;
    }
    ret.push([arr[i],temp]);
  }
  return ret;
}
// console.log(pairElement("GCG"));

function booWho(bool) {
  if (typeof bool == 'boolean') {
    return true;
  }  else {
    return false;
  }
}

// console.log(booWho(null));