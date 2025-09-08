


def load_properties(filename):
    properties = {}
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

        headers = lines[0].split(sep=',')
        for header in headers:
            properties[header] = []
        for line in lines[1:]:
            raw_values = line.split(sep=',')[1:]
            values = []
            for v in raw_values:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(0.0)   # Normalize values between 0 and 1
            min_val = min(values)
            max_val = max(values)
            if max_val != min_val:
                values = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                values = [0.0 for _ in values]

            for i, value in enumerate(values):
                #aprint(headers[i])
                try:
                    properties[headers[i]].append(float(value))
                except ValueError:
                    properties[headers[i]].append(float(0))


    return properties


properties = load_properties("properties.txt")

periodicTable = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83
} 

inversePeriodicTable = {v: k for k, v in periodicTable.items()}

def getProps(element):
    return properties[element]

# just returns a dictionary that converts the symbol into an atomic number
# here so that it doesn't take up space in the main file
def getPeriodicTableDict():

    return periodicTable

def getInversePeriodicTable(val):
    return inversePeriodicTable[val]
