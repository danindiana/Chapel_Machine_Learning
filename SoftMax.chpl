/* The Tensor class */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The Softmax class represents a Softmax activation function */
class Softmax {
  /* Forward propagation */
  proc forward(input: Tensor) {
    assert(input.shape.size == 2, "Input must be a 2-dimensional tensor");

    /* Create the output tensor */
    var output = new Tensor(input.shape);

    /* Apply the Softmax function */
    forall i in 1..input.shape[1] do
      var maxval = max reduce input.data[i, ..];
      var sumexp = + reduce exp(input.data[i, ..] - maxval);
      output.data[i, ..] = exp(input.data[i, ..] - maxval) / sumexp;

    return output;
  }
}
