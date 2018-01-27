require_relative "./configs"
require_relative "./layers/layer"
require_relative "./layers/output_layer"
require_relative "./layers/hidden_layer"

class Network
  attr_accessor :input_layer, :hidden_layers, :output_layer

  def self.optimize(training_config:)
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

  def self.predict(training_nn)
    random_input_output = NetworkConfig::TRAINING_DATA.sample
    input_to_predict = random_input_output.first
    predicted_output = random_input_output.last
    puts "\n----------------------------------------"
    puts "Calculating expected output for: #{input_to_predict}..."
    puts "----------------------------------------"

    input_layer = training_nn.input_layer
    hidden_layer = training_nn.hidden_layers.first
    output_layer = training_nn.output_layer
    hidden_weights = hidden_layer.neurons.map { |n| n.weights }
    output_weights = output_layer.neurons.map { |n| n.weights }

    config = NetworkConfig.new
    config.input = Layer.new(quantity: input_layer.neurons.length, initial_weights: [rand,rand], bias: input_layer.bias)
    config.hidden = [ HiddenLayer.new(quantity: hidden_layer.neurons.length, initial_weights: hidden_weights, bias: hidden_layer.bias)]
    config.output = OutputLayer.new(quantity: output_layer.neurons.length, initial_weights: output_weights, bias: output_layer.bias)

    prediction_nn = Network.new(config: config)
    prediction_nn.ff(input_to_predict)

    puts "-- Output Classification --"
    prediction_nn.output_layer.neurons.each_with_index { |n,i| puts "Neuron #{i+1}: #{n.output}" }
    puts "\n-- Total Error --"
    puts prediction_nn.total_error([[input_to_predict, predicted_output]])
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
  nn = Network.optimize(training_config: NetworkConfig.new)
  Network.predict(nn)
end
