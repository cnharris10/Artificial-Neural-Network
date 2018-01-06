class NetworkConfig

  attr_reader :error_threshold, :training_data, :learning_rate, :input, :hidden, :output

  TRAINING_DATA = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
  ]

  ERROR_THRESHOLD = 0.01
  LEARNING_RATE = 0.1

  def initialize
    @error_threshold = ERROR_THRESHOLD
    @training_data = TRAINING_DATA
    @learning_rate = LEARNING_RATE
    @input = InputLayerConfig.new(name: 'input')
    @hidden = [ HiddenLayerConfig.new(name: 'hidden_1') ]
    @output = OutputLayerConfig.new(name: 'output')
  end

end

class LayerConfig

  attr_reader :name, :quantity, :weights, :bias

  def initialize(name: nil, quantity:, weights:, bias:)
    @name = name
    @quantity = quantity
    @weights = weights
    @bias = bias
  end

end

class InputLayerConfig < LayerConfig

  def initialize(name:, quantity: 5, weights: nil, bias: nil)
    super(name: name, quantity: quantity, weights: weights, bias: bias)
  end

end

class HiddenLayerConfig < LayerConfig

  def initialize(name:, quantity: 5, weights: nil, bias: nil)
    super(name: name, quantity: quantity, weights: weights, bias: bias)
  end

end

class OutputLayerConfig < LayerConfig

  def initialize(name:, quantity: 1, weights: nil, bias: nil)
    super(name: name, quantity: quantity, weights: weights, bias: bias)
  end

end
