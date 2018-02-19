require 'rmagick'
require 'descriptive_statistics'
require 'byebug'

require_relative './neural_network'

include Magick

image_paths = (0..9).map { |i| "./images/numbers/training/#{i}.png" }
image_array = ImageList.new(*image_paths).inject([]) { |arr, img| arr << img.export_pixels; arr }
image_array_stats = image_array.map { |arr| { mean: arr.mean, standard_deviation: arr.standard_deviation } }
expected_outputs = image_paths.map.with_index { |image, index| arr = Array.new(10,0.0); arr[index] = 1.0; arr }

normalized_image_array = image_array.map.with_index do |input,index|
  input.map do |pixel|
    stats = image_array_stats[index]
    (pixel.to_f - stats[:mean]) / stats[:standard_deviation]
  end
end

input = N[*normalized_image_array]
output = N[*expected_outputs]
hidden_layer_size = 40
threshold = 0.0001

nn = NeuralNetwork.build(input, output, hidden_layer_size, threshold)
image_array = ImageList.new("./images/numbers/training/8-a.png").export_pixels
image_array_stats = { mean: image_array.mean, standard_deviation: image_array.standard_deviation }
normalized_image_array = image_array.map do |pixel|
  (pixel.to_f - image_array_stats[:mean]) / image_array_stats[:standard_deviation]
end

pp nn.ff(N[normalized_image_array])
