require 'matrix'
require 'nmatrix'

class NeuralNetwork
  def self.build(input, output, hidden_layer_size, threshold)
    nn = NeuralNetwork.new(input, output, hidden_layer_size, threshold)
    iteration_count = 0
    loop do
      loss = nn.determine_loss(iteration_count)
      break if loss[0] < nn.threshold
      nn.train(input, output)
      iteration_count += 1
    end
    nn
  end

  attr_reader :threshold

  def determine_loss(iteration_count)
    puts "Iteration: #{iteration_count}"
    puts "Input: #{@inputs}"
    puts "Actual Output: #{@outputs}"
    puts "Predicted Output: #{ff(@inputs)}"
    loss = ((@outputs - ff(@inputs)) ** 2).mean
    puts "Loss: #{loss}\n\n"
    loss
  end

  def initialize(inputs, outputs, hidden_size, threshold)
    @inputs = inputs
    @outputs = outputs
    @hiddenSize = hidden_size
    @threshold = threshold
    @inputSize = @inputs.row(0).size
    @outputSize = @outputs.row(0).size

    # Produce weight matrices of same size as hidden / ouput layers
    @weights = {
      hidden: NMatrix.new([@inputSize, @hiddenSize], random_weights(@inputs)),
      output: NMatrix.new([@hiddenSize, @outputSize], random_weights(@outputs))
    }
  end

  # Produce random weights for all cells matrix argument
  def random_weights(source)
    (1..source.size).to_a.map { rand }
  end

  # Apply sigmoid activation function to dot product of weights matrix
  # Apply dot producr of previous layers output to output weights
  # Apply sigmoid activation to output dot product
  def ff(input)
    dot_product = input.dot(@weights[:hidden])
    @sigmoid_product = sigmoid(dot_product)
    dot_product_of_activation = @sigmoid_product.dot(@weights[:output])
    sigmoid(dot_product_of_activation)
  end

  def sigmoid(matrix)
    matrix.map { |cell| 1.0 / (1.0 + Math::E ** -cell) }
  end

  def sigmoid_compliment(matrix)
    matrix * matrix.map { |cell| 1.0 - cell }
  end

  def fb(input, output, ff_result)
    output_delta = (output - ff_result) * sigmoid_compliment(ff_result)
    sigmoid_compliment_delta = output_delta.dot(@weights[:output].transpose) * sigmoid_compliment(@sigmoid_product)
    @weights[:hidden] += input.transpose.dot(sigmoid_compliment_delta)
    @weights[:output] += @sigmoid_product.transpose.dot(output_delta)
  end

  def train(input, output)
    fb(input, output, ff(input))
  end
end
