class MaxPool2D {
    var pool_size: int;
    var stride: int;
    
    // Constructor
    proc init(pool_size: int, stride: int) {
        assert(pool_size > 0, "Pool size must be greater than 0");
        assert(stride > 0, "Stride must be greater than 0");
        this.pool_size = pool_size;
        this.stride = stride;
    }
    
    // Forward propagation
    proc forward(input: [?D] real) {
        assert(D.rank == 4, "Input must be 4-dimensional");
        assert(input.eltType == real, "Input elements must be real numbers");
        
        var (n, h, w, c) = D.dims(); // number of images, height, width, channels
        
        assert(this.pool_size <= h && this.pool_size <= w, 
               "Pool size must not exceed input dimensions");
        assert(this.stride <= h && this.stride <= w, 
               "Stride must not exceed input dimensions");
        
        var output_height = (h - this.pool_size) / this.stride + 1;
        var output_width = (w - this.pool_size) / this.stride + 1;
        
        assert(output_height > 0 && output_width > 0, 
               "Pooling operation results in empty output dimensions");
        
        var output: [1..n, 1..output_height, 1..output_width, 1..c] real;
        
        forall i in 1..n do
            for j in 1..output_height by this.stride do
                for k in 1..output_width by this.stride do
                    for l in 1..c do
                        output[i, j, k, l] = maxPool(input[i, j..#this.pool_size, k..#this.pool_size, l]);
        return output;
    }
    
    // Max pooling operation
    proc maxPool(input_patch: [?D] real) {
        assert(D.rank == 2, "Input patch to maxPool must be 2-dimensional");
        assert(input_patch.eltType == real, "Input patch elements must be real numbers");
        return max reduce input_patch;
    }
}
