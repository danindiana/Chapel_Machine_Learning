/* The Tensor class */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The ReLU class represents a ReLU activation function */
class ReLU {
  /* Forward propagation */
  proc forward(input: Tensor) {
    assert(input.shape.size == 2, "Input must be a 2-dimensional tensor");

    /* Create the output tensor */
    var output = new Tensor(input.shape);

    /* Apply the ReLU function */
    forall i in output.data.domain do
      output.data[i] = max(0.0, input.data[i]);

    return output;
  }
}
