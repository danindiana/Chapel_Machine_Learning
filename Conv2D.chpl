/* The Tensor class will be used to represent inputs, weights, biases, and outputs */
class Tensor {
  var shape: [1..4] int;
  var data: [1..shape[1], 1..shape[2], 1..shape[3], 1..shape[4]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2], 1..shape[3], 1..shape[4]};
  }
}

/* The Conv2D class represents a 2D convolutional layer */
class Conv2D {
  var kernelSize: int;
  var numFilters: int;
  var stride: int;
  var pad: int;
  var weights: Tensor;
  var biases: Tensor;

  /* Constructor */
  proc init(numFilters: int, kernelSize: int, stride: int = 1, pad: int = 0) {
    this.numFilters = numFilters;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.pad = pad;
    this.weights = new Tensor([numFilters, kernelSize, kernelSize, 1]);
    this.biases = new Tensor([numFilters]);
  }

  /* Forward propagation */
  proc forward(input: Tensor) {
    assert(input.shape.size == 4, "Input must be a 4-dimensional tensor");
    assert(this.weights.shape[2:3] == [this.kernelSize, this.kernelSize],
           "Weights must be of size [numFilters, kernelSize, kernelSize, 1]");
    assert(this.biases.shape.size == 1 && this.biases.shape[1] == this.numFilters,
           "Biases must be of size [numFilters]");

    /* Calculate the output shape */
    var outputHeight = (input.shape[2] - this.kernelSize + 2*this.pad) / this.stride + 1;
    var outputWidth = (input.shape[3] - this.kernelSize + 2*this.pad) / this.stride + 1;

    /* Create the output tensor */
    var output = new Tensor([input.shape[1], outputHeight, outputWidth, this.numFilters]);

    /* Perform the convolution operation */
    // TODO: Add the actual convolution operation

    return output;
  }
}
