from predict_material import predict_material
# Example usage:
spectral_data = [215.9, 607.81, 1899.83, 175.62, 968.77, 550.97, 41.88, 91.15, 394.43, 
                 107.92, 154.77, 24.58, 40.62, 29.73, 85.16, 141.69, 35.12, 27.1]

filament_color = 'R'  # Example: 'K' for Black

predicted_material = predict_material(spectral_data, filament_color)
print(f"Predicted Filament Material: {predicted_material}")
