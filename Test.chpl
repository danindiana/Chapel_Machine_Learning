module Test {

  use Model;
  use DataLoader;
  use Utils;

  var model: Model;
  var testLoader: DataLoader;

  proc init(model: Model, testLoader: DataLoader) {
    this.model = model;
    this.testLoader = testLoader;
  }

  proc test() {
    this.model.eval();
    var total: int = 0;
    var correct: int = 0;

    for (batchData, batchLabels) in this.testLoader {
      var outputs = this.model.forward(batchData);
      var predicted = maxloc(outputs, 1).second;
      
      total += batchLabels.size;
      correct += + reduce (predicted == batchLabels);
    }

    var accuracy = correct / total: real;
    writeln("Accuracy on test set: ", accuracy);
  }
}
