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
  var momentum: real;
  var velocity: Tensor;
  var decay: real;

  /* Constructor */
  proc init(learningRate: real = 0.01, momentum: real = 0.9, decay: real = 0.0) {
    super.init(learningRate);
    this.momentum = momentum;
    this.decay = decay;
  }

  /* Update function */
  proc update(params: Tensor, grads: Tensor) {
    assert(params.shape.size == 2, "Parameters must be a 2-dimensional tensor");
    assert(grads.shape.size == 2, "Gradients must be a 2-dimensional tensor");
    assert(params.shape == grads.shape, "Parameters and gradients must be the same shape");

    /* Initialize velocity if it's not already initialized */
    if this.velocity == nil then
      this.velocity = new Tensor(params.shape);

    /* Update the velocity and parameters */
    forall (v, p, g) in zip(this.velocity.data, params.data, grads.data) do {
      v = this.momentum * v - this.learningRate * g;
      p += v;
    }

    /* Decay the learning rate */
    this.learningRate *= (1.0 - this.decay);
  }
}
