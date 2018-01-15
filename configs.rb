class NetworkConfig

  attr_reader :error_threshold, :training_data, :learning_rate
  attr_accessor :input, :hidden, :output

  TRAINING_DATA = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
  ]

  ERROR_THRESHOLD = 0.001
  LEARNING_RATE = 0.1

  def initialize(error_threshold: ERROR_THRESHOLD, training_data: TRAINING_DATA, learning_rate: LEARNING_RATE,
    input: Layer.new(quantity: 2), hidden: [ HiddenLayer.new(quantity: 5) ], output: OutputLayer.new(quantity: 1))
    @error_threshold = error_threshold
    @training_data = training_data
    @learning_rate = learning_rate
    @input = input
    @hidden = hidden
    @output = output
  end

end

class LayerConfig

  attr_reader :name, :quantity, :weights, :bias

  def initialize(name: nil, quantity:, initial_weights:, bias:)
    @name = name
    @quantity = quantity
    @weights = weights
    @bias = bias
  end

end

class InputLayerConfig < LayerConfig

  def initialize(name: 'input', quantity: 2, initial_weights: nil, bias: nil)
    super(name: name, quantity: quantity, initial_weights: initial_weights, bias: bias)
  end

end

class HiddenLayerConfig < LayerConfig

  def initialize(name: 'hidden', quantity: 5, initial_weights: nil, bias: nil)
    super(name: name, quantity: quantity, initial_weights: initial_weights, bias: bias)
  end

end

class OutputLayerConfig < LayerConfig

  def initialize(name: 'output', quantity: 1, initial_weights: nil, bias: nil)
    super(name: name, quantity: quantity, initial_weights: initial_weights, bias: bias)
  end

end
