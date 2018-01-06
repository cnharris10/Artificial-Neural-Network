require_relative "./configs"
require_relative "./layers/layer"
require_relative "./layers/output_layer"
require_relative "./layers/hidden_layer"
require 'byebug'

class Network
  attr_accessor :input_layer, :hidden_layers, :output_layer

  def self.run(config)
    count, training = 0, true
    nn = Network.new(config: config)
    while training do
      input, output = config.training_data.sample
      nn.train(input, output)
      error = nn.total_error(config.training_data)
      puts "#{count}: #{error} - Inputs: #{input}, Output: #{output}"
      count += 1
      training = false if error < config.error_threshold
    end

    puts "\n----------------------------------------"
    puts "Final weights (after #{count} runs)"
    puts "----------------------------------------"
    puts "-- Hidden Layer --"
    nn.hidden_layers.first.neurons.each_with_index { |n,i| puts "Neuron #{i+1}: #{n.weights}" }
    puts "\n-- Output Layer --"
    nn.output_layer.neurons.each_with_index { |n,i| puts "Neuron #{i+1}: #{n.weights}" }
    puts "----------------------------------------\n\n"
    nn
  end

  def initialize(config:)
    raise "Must supply a NetworkConfig instance for `config`" unless config.is_a?(NetworkConfig)
    @config = config
    build_layers
    @hidden_layers.first.initialize_weights(@config.training_data[0][0].length, true)
    @output_layer.initialize_weights(@hidden_layers.first)
  end

  def build_layers
    @input_layer = Layer.new(@config.input.quantity)
    ### Must account for multiple hidden layers ####
    @hidden_layers = [ HiddenLayer.new(@config.hidden[0].quantity) ]
    @output_layer = OutputLayer.new(@config.output.quantity)
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

Network.run(NetworkConfig.new) if $0 == 'network.rb'
