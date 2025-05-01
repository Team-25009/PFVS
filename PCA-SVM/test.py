# demo_print.py
from predict_material import predict_material

# 1) Red PLA
scan_pla_red = [
    726, 136, 322, 117, 207, 271, 110, 220,
    153, 210, 45, 71, 21, 14, 23, 72, 30, 19
]
print("PLA (red) →", predict_material(scan_pla_red, 'red'))

# 2) White PET
scan_pet_white = [
    3461, 991, 2294, 1029, 1458, 1636, 696, 801,
    163, 457, 48, 130, 22, 15, 24, 76, 61, 38
]
print("PET (white) →", predict_material(scan_pet_white, 'white'))

# 3) Blue ASA
scan_asa_blue = [
    1050, 260, 655, 211, 277, 304, 115, 105,
    139, 71, 43, 52, 20, 14, 22, 72, 20, 17
]
print("ASA (blue) →", predict_material(scan_asa_blue, 'blue'))
