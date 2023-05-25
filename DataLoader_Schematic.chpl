use CyclicDist;

class DataLoader {
  var dataset: Dataset;
  var batchSize: int;
  var shuffle: bool;
  var seed: int;
  var numWorkers: int;

  proc init(dataset: Dataset, batchSize: int, shuffle: bool, seed: int, numWorkers: int) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.seed = seed;
    this.numWorkers = numWorkers;
  }

  iter these() ref {
    if this.shuffle {
      this.dataset.data, this.dataset.labels = shuffle(this.dataset.data, this.dataset.labels, this.seed);
    }
    coforall w in 0..#this.numWorkers with (var q = new BlockCyclic(startIdx=(w*this.batchSize)+1, blocksize=this.batchSize, targetLocales=Locales)) {
      for b in w*this.batchSize..#this.dataset.data.size by this.numWorkers*this.batchSize {
        var batchData = this.dataset.data[b..#this.batchSize];
        var batchLabels = this.dataset.labels[b..#this.batchSize];
        // Perform data augmentation here (e.g., random cropping, flipping, normalization, etc.)
        yield (batchData, batchLabels);
      }
    }
  }
}
