require_relative "../neuron"

class Layer

  attr_accessor :quantity, :initial_weights, :bias, :neurons

  def initialize(quantity:, initial_weights: nil, bias: rand)
    # Every neuron in a layer shares the same bias
    @quantity = quantity
    @initial_weights = initial_weights
    @bias = bias
    @neurons = []
    create_neurons
  end

  def create_neurons
    @quantity.times { |i| @neurons << Neuron.new(@bias) }
  end

  def initialize_weights(obj, from_inputs = false)
    from_inputs ? initialize_weights_from_inputs(obj) : initialize_weights_from_previous_layer(obj)
  end

  def initialize_weights_from_inputs(inputs)
    weight_num = 0
    @neurons.each do |h_neuron|
      inputs.times do |i|
        h_neuron.weights.push(@initial_weights ? @initial_weights[weight_num][i] : rand)
      end

      # Revisit this line
      h_neuron.weights = h_neuron.weights.flatten.compact
      weight_num += 1
    end
  end

  def initialize_weights_from_previous_layer(layer, from_inputs = false)
    weight_num = 0
    @neurons.each_with_index do |o_neuron, o_index|
      layer.neurons.each_with_index do |h_neuron, index|
        o_neuron.weights.push(@initial_weights ? @initial_weights[weight_num] : rand)
        weight_num += 1
      end

      # Revisit this line
      o_neuron.weights = o_neuron.weights.flatten.compact
    end
  end

  def ff(inputs)
    @neurons.map { |neuron| neuron.calculate_output(inputs) }
  end

end
