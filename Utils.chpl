module Utils {

  use Random;
  use LinearAlgebra;

  /* Shuffle two arrays in-place using the given seed. */
  proc shuffle(ref arr1: [?D], ref arr2: [D], seed: int) {
    var rng = new RandomStream(int, seed=seed);
    var indices = D.indices;
    rng.shuffle(indices);
    arr1 = arr1[indices];
    arr2 = arr2[indices];
  }

  /* Split an array into a training set and a test set with the given ratio. */
  proc trainTestSplit(arr: [?D], ratio: real): 2*arr.type {
    var rng = new RandomStream();
    var indices = D.indices;
    rng.shuffle(indices);
    var splitIdx = round(ratio * D.size): int;
    return (arr[indices[1..splitIdx]], arr[indices[splitIdx+1..]]);
  }

  /* Compute the accuracy of the predicted labels against the true labels. */
  proc accuracy(yPred: [?D], yTrue: [D]): real {
    return + reduce (yPred == yTrue) / D.size:real;
  }

  /* Standardize an array using mean normalization. */
  proc standardize(ref arr: [?D]) {
    var mean = + reduce arr / D.size:real;
    var std = sqrt((+ reduce (arr - mean)**2) / D.size:real);
    arr -= mean;
    arr /= std;
  }

}
