require_relative "../neuron"

class Layer

  attr_accessor :quantity, :bias, :neurons

  def initialize(quantity, bias = rand)
    # Every neuron in a layer shares the same bias
    @quantity = quantity
    @bias = bias
    @neurons = []
    @initial_weights = nil
    create_neurons
  end

  def create_neurons
    @quantity.times { |i| @neurons << Neuron.new(@bias) }
  end

  def initialize_weights(obj, from_inputs = false)
    from_inputs ? initialize_weights_from_inputs(obj) : initialize_weights_from_previous_layer(obj)
  end

  def initialize_weights_from_inputs(inputs, from_inputs = false)
    weight_num = 0
    @neurons.each do |h_neuron|
      inputs.times do |i|
        h_neuron.weights.push(@initial_weights ? @initial_weights[weight_num] : rand)
        weight_num += 1
      end
    end
  end

  def initialize_weights_from_previous_layer(layer, from_inputs = false)
    weight_num = 0
    @neurons.each do |o_neuron|
      layer.neurons.each do |h_neuron|
        o_neuron.weights.push(@initial_weights ? @initial_weights[weight_num] : rand)
        weight_num += 1
      end
    end
  end

  def ff(inputs)
    @neurons.map { |neuron| neuron.calculate_output(inputs) }
  end

end
