import numpy as np
import itertools
import util
from GCN import GRAPH_SIZE

def process_poscar(poscar_file):

    defect_list = []
    cord_list = []
    #extracts hse and pbe cords
    with open(poscar_file, 'r') as poscarRead:
        #skip to line that has 'Direct' in its name
        poscarLines = poscarRead.readlines()

        # gets defect elements
        poscarIndex = 5
        elements = poscarLines[poscarIndex].strip().split()

        # gets amount of each element
        poscarIndex = 6
        amounts = [int(x) for x in poscarLines[poscarIndex].strip().split()]
        for n in range(0, len(elements)):
            if elements[n] != "Si":
                for i in range(0, amounts[n]):
                    defect_list.append(elements[n])

        while poscarLines[poscarIndex].strip() != "Direct":   poscarIndex += 1

        #skip the next line
        poscarIndex += 1

        #extract the coordinates
        while True:
            #goes until no more cords to process
            if poscarLines[poscarIndex].strip() == "":    break
            cord_list.append(np.array([float(x) for x in poscarLines[poscarIndex].split()]))
            poscarIndex += 1


        # for these values if they are over 0.5, subtract 1
        for n in range(0, len(cord_list)):
            for i in range(0, 3):
                if cord_list[n][i] > 0.5:
                    cord_list[n][i] -= 1
    
    return get_filtered_cords(np.array(cord_list), defect_list), defect_list


def get_filtered_cords(cords, defects):

    minMagnitude = lambda aList, b: min(np.linalg.norm(a - b) for a in aList)

    filteredCordsList = np.zeros((GRAPH_SIZE, 3))

    centerCords = []
    for n in range(len(defects)):
        centerCords.append(cords[n])

    expanded_cords = np.zeros(((len(cords) - len(centerCords)) * 27, 3))

    # Just brute force expanding the list of atoms so that it adds in the repeat sorroundings
    flat_index = 0
    for exIndex in range(len(centerCords), len(cords)):
        for shift in itertools.product([-1, 0, 1], repeat=3):
            expanded_cords[flat_index] = cords[exIndex] + np.array(shift)
            flat_index += 1

    # Sort expanded lists by distance to the center coordinates
    expanded_cords = expanded_cords[np.argsort([minMagnitude(centerCords, x) for x in expanded_cords])]


    #creates the filtered list
    filteredCordsList = np.append(expanded_cords[:len(centerCords)], expanded_cords[:GRAPH_SIZE-len(centerCords)], axis=0)
    return filteredCordsList

def create_adjacency_matrix(cords, threshold=0.25):
    num_atoms = cords.shape[0]
    adj_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            adj_matrix[i, j] = np.linalg.norm(cords[i] - cords[j])
            adj_matrix[j, i] = adj_matrix[i, j]

    adj_matrix = np.where(adj_matrix < threshold, (threshold - adj_matrix)/threshold, 0)

    return adj_matrix

def getFeatureVector(cords, defects, charge):

    feature_vector = np.zeros((len(cords), 20))  # 20 features per atom

    for i in range(len(defects)):
        if i < len(defects):
            feature_vector[i, :19] = util.getProps(defects[i])
        else:
            feature_vector[i, :19] = util.getProps('Si')
        feature_vector[i, 19] = charge

    return feature_vector

def get_graph_data(poscar_file, charge):

    cords, defects = process_poscar(poscar_file)
    adj_matrix = create_adjacency_matrix(cords)
    feature_vector = getFeatureVector(cords, defects, charge)

    return adj_matrix, feature_vector