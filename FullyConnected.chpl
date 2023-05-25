/* The Tensor class, similar to previous one */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The FullyConnected class represents a fully connected layer */
class FullyConnected {
  var inputSize: int;
  var outputSize: int;
  var weights: Tensor;
  var biases: Tensor;

  /* Constructor */
  proc init(inputSize: int, outputSize: int) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = new Tensor([inputSize, outputSize]);
    this.biases = new Tensor([outputSize]);
  }

  /* Forward propagation */
  proc forward(input: Tensor) {
    assert(input.shape.size == 2, "Input must be a 2-dimensional tensor");
    assert(this.weights.shape == [this.inputSize, this.outputSize],
           "Weights must be of size [inputSize, outputSize]");
    assert(this.biases.shape.size == 1 && this.biases.shape[1] == this.outputSize,
           "Biases must be of size [outputSize]");

    /* Calculate the output */
    var output = new Tensor([input.shape[1], this.outputSize]);

    /* Perform the matrix multiplication and addition */
    // TODO: Add the actual computation of output

    return output;
  }
}
