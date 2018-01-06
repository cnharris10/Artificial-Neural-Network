class Neuron
  attr_accessor :bias, :inputs, :weights

  def initialize(bias = nil)
    @bias = bias
    @weights = []
  end

  def calculate_output(inputs)
    @inputs = inputs
    @output = activation_function(total_net_input)
  end

  def total_net_input
    total = 0
    @inputs.each_with_index do |input, index|
      total += (input * @weights[index])
    end
    total + @bias
  end

  # Sigmoid logistic regression
  def activation_function(total_net_input)
    1 / (1 + Math::E ** -total_net_input)
  end

  # Partial derivative error in reference to input
  # See https://en.wikipedia.org/wiki/Backpropagation
  def partial_derivitive_error(target_output)
    -(target_output - @output) * complementing_output_product
  end

  # Determining the variance of expected vs actual outputs
  def mean_squared_error(target_output)
    0.5 * (target_output - @output) ** 2
  end

  # Partial derivative for determining output accurracy in reference to inputs
  def complementing_output_product
    @output * (1 - @output)
  end

end
