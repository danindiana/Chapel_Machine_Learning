/* The Tensor class */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The CrossEntropyLoss class represents a cross-entropy loss function */
class CrossEntropyLoss {
  /* Forward propagation */
  proc forward(pred: Tensor, target: Tensor) {
    assert(pred.shape.size == 2, "Prediction must be a 2-dimensional tensor");
    assert(target.shape.size == 2, "Target must be a 2-dimensional tensor");
    assert(pred.shape == target.shape, "Prediction and target must be the same shape");

    /* Compute the cross-entropy loss */
    var loss = 0.0;
    forall (p, t) in zip(pred.data, target.data) do
      loss -= t * log(p);

    return loss / pred.shape[1];  // Normalize by the number of instances
  }
}
