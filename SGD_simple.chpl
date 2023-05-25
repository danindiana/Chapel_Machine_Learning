/* The Tensor class */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The Optimizer class represents a basic optimizer */
class Optimizer {
  var learningRate: real;

  /* Constructor */
  proc init(learningRate: real = 0.01) {
    this.learningRate = learningRate;
  }

  /* Update function to be implemented by subclasses */
  proc update(params: Tensor, grads: Tensor) {
    // Placeholder
  }
}

/* The SGD class represents a Stochastic Gradient Descent optimizer */
class SGD: Optimizer {
  /* Constructor */
  proc init(learningRate: real = 0.01) {
    super.init(learningRate);
  }

  /* Update function */
  proc update(params: Tensor, grads: Tensor) {
    assert(params.shape.size == 2, "Parameters must be a 2-dimensional tensor");
    assert(grads.shape.size == 2, "Gradients must be a 2-dimensional tensor");
    assert(params.shape == grads.shape, "Parameters and gradients must be the same shape");

    /* Update the parameters */
    forall (p, g) in zip(params.data, grads.data) do
      p -= this.learningRate * g;
  }
}
