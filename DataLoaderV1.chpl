use IO;
use List;
use Random;

class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

class Dataset {
  var data: [1..0] Tensor;
  var labels: [1..0] Tensor;

  proc init(data: [?D] Tensor, labels: [?D] Tensor) {
    this.data = data;
    this.labels = labels;
  }
}

class DataLoader {
  var dataset: Dataset;
  var batchSize: int;
  var shuffle: bool;
  var seed: int;

  proc init(dataset: Dataset, batchSize: int, shuffle: bool, seed: int) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.seed = seed;
  }

  iter these() ref {
    if this.shuffle {
      this.dataset.data, this.dataset.labels = shuffle(this.dataset.data, this.dataset.labels, this.seed);
    }
    for b in 0..#this.dataset.data.size by this.batchSize {
      var batchData = this.dataset.data[b..#this.batchSize];
      var batchLabels = this.dataset.labels[b..#this.batchSize];
      yield (batchData, batchLabels);
    }
  }
}

proc shuffle(data: [?D] Tensor, labels: [?D] Tensor, seed: int) {
  var rng = new RandomStream(int, seed=seed);
  var indices = [i in 1..data.size] i;
  rng.shuffle(indices);
  return (data[indices], labels[indices]);
}
