require_relative "./layer"

class OutputLayer < Layer

  def train(outputs)
    arr = []
    @neurons.each_with_index do |o_neuron, o_index|
      expected_output = outputs[o_index]
      arr << o_neuron.partial_derivitive_error(expected_output)
    end
    arr
  end

  def update_weights(output_layer_neuron_deltas, learning_rate)
    @neurons.each_with_index do |o_neuron, o_index|
      o_neuron.weights.each_with_index do |weight, w_index|
        pd_error_wrt_weight = output_layer_neuron_deltas[o_index] * o_neuron.inputs[w_index]
        o_neuron.weights[w_index] -= learning_rate * pd_error_wrt_weight
      end
    end
  end

end
