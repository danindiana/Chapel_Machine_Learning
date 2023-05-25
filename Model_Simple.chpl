/* The Tensor class */
class Tensor {
  var shape: [1..2] int;
  var data: [1..shape[1], 1..shape[2]] real;

  proc init(shape: [?D] int) {
    this.shape = shape;
    this.data = {1..shape[1], 1..shape[2]};
  }
}

/* The Layer class represents a basic neural network layer */
class Layer {
  /* Forward propagation function to be implemented by subclasses */
  proc forward(input: Tensor) returns Tensor {
    // Placeholder
    return new Tensor(input.shape);
  }

  /* Backward propagation function to be implemented by subclasses */
  proc backward(gradOutput: Tensor) returns Tensor {
    // Placeholder
    return new Tensor(gradOutput.shape);
  }
}

/* The Model class represents a neural network model */
class Model {
  var layers: [1..0] Layer;

  /* Constructor */
  proc init() {
    this.layers = [1..0] Layer;
  }

  /* Add a layer to the model */
  proc addLayer(layer: Layer) {
    this.layers.push_back(layer);
  }

  /* Forward propagation */
  proc forward(input: Tensor) returns Tensor {
    var output = input;
    for layer in this.layers do
      output = layer.forward(output);
    return output;
  }

  /* Backward propagation */
  proc backward(gradOutput: Tensor) returns Tensor {
    var gradInput = gradOutput;
    for layer in this.layers by -1 do
      gradInput = layer.backward(gradInput);
    return gradInput;
  }
}
