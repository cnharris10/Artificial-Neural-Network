require 'matrix'
require 'nmatrix'
require "byebug"

class NeuralNetwork
  def initialize(inputs, outputs, hidden_size)
    @inputs = inputs
    @outputs = outputs
    @inputSize = @inputs.row(0).size
    @outputSize = @outputs.row(0).size
    @hiddenSize = hidden_size

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

input = N[[0.66666667, 1.0, 0.89, 0.45], [0.33333333, 0.55555556, 0.76, 0.81], [1.0, 0.66666667, 0.47, 0.68]]
output = N[[0.92], [0.86], [0.89]]
nn = NeuralNetwork.new(input, output, 4)
iteration_count = 0
loop do
  puts "Iteration: #{iteration_count}"
  puts "Input: #{input}"
  puts "Actual Output: #{output}"
  puts "Predicted Output: #{nn.ff(input)}"
  loss = ((output - nn.ff(input)) ** 2).mean
  puts "Loss: #{loss}\n\n" # mean sum squared loss
  break if loss[0] < 0.0001
  nn.train(input, output)
  iteration_count += 1
end
