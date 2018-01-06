require_relative "./layer"

class HiddenLayer < Layer

  def train(output_layer, output_layer_neuron_deltas)
    hidden_layer_neuron_deltas = []
    @neurons.each_with_index do |neuron, n_index|
      error = 0
      output_layer.neurons.each_with_index do |o_neuron, o_index|
        error += output_layer_neuron_deltas[o_index] * o_neuron.weights[n_index]
      end
      hidden_layer_neuron_deltas << error * neuron.complementing_output_product
    end
    hidden_layer_neuron_deltas
  end

  def update_weights(hidden_layer_neuron_deltas, learning_rate)
    @neurons.each_with_index do |neuron, n_index|
      neuron.weights.each_with_index do |weight, w_index|
        pd_error_wrt_weight = hidden_layer_neuron_deltas[n_index] * neuron.inputs[w_index]
        neuron.weights[w_index] -= learning_rate * pd_error_wrt_weight
      end
    end
  end

end
