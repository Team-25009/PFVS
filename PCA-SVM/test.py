from predict_material_PLSDA import predict_material_plsda

# put your 18 spectral readings into a list
values = [
    1159, 537, 1490, 616, 684, 479, 153, 143,
     145,  86,   41,  37,  21,  14,  29,  82,
      32,  23
]

# now call with (spectral_data, color_label)
material = predict_material_plsda(values, 'Blue')
print(material)
