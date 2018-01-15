require_relative "./configs"
require_relative "./layers/layer"
require_relative "./layers/output_layer"
require_relative "./layers/hidden_layer"

class Network
  attr_accessor :input_layer, :hidden_layers, :output_layer

  def self.run(training_config: training_config)
    count, training = 0, true
    nn = Network.new(config: training_config)
    puts "\n----------------------------------------"
    puts "Training..."
    puts "----------------------------------------"
    while training do
      input, output = training_config.training_data.sample
      nn.train(input, output)
      error = nn.total_error(training_config.training_data)
      puts "Run: #{count} - Error: #{error} - Inputs: #{input}, Output: #{output}"
      count += 1
      training = false if error < training_config.error_threshold
    end

    puts "\n----------------------------------------"
    puts "Final training weights (after #{count} runs)"
    puts "----------------------------------------"
    puts "-- Input Layer --"
    nn.input_layer.neurons.each_with_index { |n,i| puts "Neuron #{i+1} - bias: #{n.bias}" }
    puts "\n-- Hidden Layer --"
    nn.hidden_layers.first.neurons.each_with_index { |n,i| puts "Neuron #{i+1} - weights: #{n.weights} - bias: #{n.bias}" }
    puts "\n-- Output Layer --"
    nn.output_layer.neurons.each_with_index { |n,i| puts "Neuron #{i+1} - weights: #{n.weights} - bias: #{n.bias}" }
    puts "----------------------------------------\n\n"
    nn
  end

  def self.calculate(nn)
    input_data = [1,1]
    puts "\n----------------------------------------"
    puts "Calculating expected output for: #{input_data}..."
    puts "----------------------------------------"

    hidden_weights = nn.hidden_layers.first.neurons.map { |n| n.weights }
    output_weights = nn.output_layer.neurons.map { |n| n.weights }
    config = NetworkConfig.new
    config.input = Layer.new(quantity: 2, initial_weights: [rand,rand], bias: nn.input_layer.bias)
    config.hidden = [ HiddenLayer.new(quantity: 5, initial_weights: hidden_weights, bias: nn.hidden_layers.first.bias)]
    config.output = OutputLayer.new(quantity: 1, initial_weights: output_weights, bias: nn.output_layer.bias)
    nn2 = Network.new(config: config)
    nn2.ff([1,1])

    puts "-- Output Classification --"
    nn2.output_layer.neurons.each_with_index { |n,i| puts "Neuron #{i+1}: #{n.output}" }
    puts "\n-- Total Error --"
    puts nn2.total_error([[input_data,[0]]])
    puts "----------------------------------------\n\n"
  end

  def initialize(config:)
    raise "Must supply a NetworkConfig instance for `config`" unless config.is_a?(NetworkConfig)
    @config = config
    build_layers
    @hidden_layers.first.initialize_weights(@config.training_data[0][0].length, true)
    @output_layer.initialize_weights(@hidden_layers.first)
  end

  def build_layers
    @input_layer = @config.input || Layer.new(quantity: @config.input.quantity)
    ### Must account for multiple hidden layers ####
    @hidden_layers = @config.hidden || [ HiddenLayer.new(quantity: @config.hidden[0].quantity) ]
    @output_layer = @config.output || OutputLayer.new(quantity: @config.output.quantity)
  end

  def ff(inputs)
    result = @hidden_layers.first.ff(inputs)
    @output_layer.ff(result)
  end

  def train(inputs, outputs)
    ff(inputs)

    # Compare output layer neuron outputs to expected outputs
    output_layer_neuron_deltas = @output_layer.train(outputs)

    # 2. Hidden neuron deltas
    ### Must account for multiple hidden layers ####
    hidden_layer_neuron_deltas = @hidden_layers.first.train(@output_layer, output_layer_neuron_deltas)

    # 3. Update output neuron weights
    @output_layer.update_weights(output_layer_neuron_deltas, @config.learning_rate)

    # 4. Update hidden neuron weights
    ### Must account for multiple hidden layers ####
    @hidden_layers.first.update_weights(hidden_layer_neuron_deltas, @config.learning_rate)
  end

  def total_error(training_data)
    error = 0
    training_data.each do |training_data_pair|
      inputs, outputs = training_data_pair
      ff(inputs)
      outputs.each_with_index do |output, index|
        error += @output_layer.neurons[index].mean_squared_error(output)
      end
    end
    error
  end
end

if $0 == 'network.rb'
  nn = Network.run(training_config: NetworkConfig.new)
  Network.calculate(nn)
end
