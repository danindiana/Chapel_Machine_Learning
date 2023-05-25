module Train {

  use Model;
  use DataLoader;
  use Optimizer;
  use LossFunction;
  use Utils;

  var model: Model;
  var dataloader: DataLoader;
  var optimizer: Optimizer;
  var lossFunction: LossFunction;
  var numEpochs: int;

  proc init(model: Model, dataloader: DataLoader, optimizer: Optimizer, lossFunction: LossFunction, numEpochs: int) {
    this.model = model;
    this.dataloader = dataloader;
    this.optimizer = optimizer;
    this.lossFunction = lossFunction;
    this.numEpochs = numEpochs;
  }

  proc train() {
    for epoch in 1..this.numEpochs {
      this.model.train();
      for (batchData, batchLabels) in this.dataloader {
        this.optimizer.zeroGrad();
        var outputs = this.model.forward(batchData);
        var loss = this.lossFunction.forward(outputs, batchLabels);
        loss.backward();
        this.optimizer.step();
      }
      this.model.eval();
      // Compute validation loss and accuracy here, if necessary.
    }
  }

}
