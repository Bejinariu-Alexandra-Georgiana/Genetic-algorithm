import numpy as np
from scipy.spatial.distance import euclidean

import re


def extract_homo_lumo(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    homo = None
    lumo = None

    # extract homo 
    for line in reversed(lines):
        if " Alpha  occ. eigenvalues --" in line:
            values = line.split()[4:]
            homo = float(values[-1])
            break


    # extract lumo 
    for line in lines:
        if " Alpha virt. eigenvalues --" in line:
            values = line.split()[4:]
            lumo = float(values[0])
            break
    
    return homo, lumo


class VecR3:
   def __init__(self, x = 0e0, y = 0e0, z = 0e0):
      self.x = x; self.y = y; self.z = z    

class Atom:
    def __init__(self, name =  '', resi = '', segm = '', ires = '', r = None, occp= '', beta= '', symb= ''):
        if r is None:
            r = VecR3()
        self.name = name
        self.resi = resi
        self.segm = segm
        self.ires = ires
        self.r = r
        self.occp = occp
        self.beta = beta
        self.symb = symb


##### Find the coordinates 

def molCoordHex(file):
    with open(file, 'r') as f:
        data = f.readlines()

    # extract the atom coordinates
    coords = []
    for line in data:
        try:
            if re.match(r'^HETATM', line):
                coords.append(list(map(float, line.split()[4:7])))
        except ValueError:
            pass

    # loop through all possible hexagons and check if they are valid
    triangles = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            for k in range(j+1, len(coords)):
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[j]
                x3, y3, z3 = coords[k]
                dist12 = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
                dist13 = ((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2)**0.5
                dist23 = ((x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2)**0.5
                if abs(dist12 - dist13) < 0.1 and abs(dist13 - dist23) < 0.1:
                    triangles.append([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])

    if len(triangles) > 0:
        # return the first triangle found 
        triangle = np.array(triangles[0])
        return triangle 
    else:
        return None




# Define the vertices of the two triangles
triangle_1 = molCoordHex('molA.txt')
triangle_2= molCoordHex('btt.txt')
initial_triangle_2 = molCoordHex('btt.txt')

    # Genetic algorithm parameters
POPULATION_SIZE = 1000  # Number of individuals in the population
MUTATION_RATE = 0.01  # Probability of a gene mutating
GENERATIONS = 500  # Number of generations

donor_homo, donor_lumo = extract_homo_lumo('btt opt.out')
acceptor_homo, acceptor_lumo = extract_homo_lumo('C60 opt.out')

def genetic_algorithm():
    def calculate_fitness(population, donor_homo, donor_lumo, acceptor_lumo, acceptor_homo):
        fitness_scores = []
        for individual in population:
            # The new position of triangle_2 
            displacement_vector = individual.reshape((3, 3))
            new_triangle_2 = triangle_2 + displacement_vector

            # Sum of the squared distances between the vertices of the two triangles
            distances = np.sum((triangle_1 - new_triangle_2)**2)

            # Extract the homo and lumo
            btt_homo = donor_homo
            c60_lumo = acceptor_lumo
            btt_lumo = donor_lumo
            c60_homo = acceptor_homo

            # Calculate the LUMO-HOMO gaps
            charge_transfer = c60_lumo - btt_homo
            donor_diff =  btt_lumo - btt_homo
            acceptor_diff =  c60_lumo - c60_homo
    
            # If the HOMO-LUMO energy difference is positive for both molecules, charge transfer is possible
            if charge_transfer > 0 and donor_diff > 0 and acceptor_diff > 0:
            # Add the penalty term to the fitness score with a weight factor
                fitness_scores.append(distances + 0.1* abs(charge_transfer))
            else:
            # If charge transfer is not possible, docking can not be performed
                fitness_scores.append(1e6)
        return np.array(fitness_scores)

    def crossover(parent_1, parent_2):
        # Single-point crossover.
        
        # Choose a random crossover point
        crossover_point = np.random.randint(1, len(parent_1))


        child_1 = np.concatenate([parent_1[:crossover_point], parent_2[crossover_point:]])
        child_2 = np.concatenate([parent_2[:crossover_point], parent_1[crossover_point:]])

        return child_1, child_2

    def mutate(individual):
         for i in range(len(individual)):
            # 0.1 - flexibility factor
            if np.random.rand() < MUTATION_RATE:
                individual[i] += np.random.randn() * 0.1
        return individual


    # Initial population
    population = np.random.randn(POPULATION_SIZE, 9)

    # Run the genetic algorithm 
    for generation in range(GENERATIONS):
        # Fitness scores for the population
        fitness_scores = calculate_fitness(population, donor_homo, donor_lumo, acceptor_lumo, acceptor_homo)

        # Tournament selection
        parent_indices = []
        for i in range(POPULATION_SIZE):
            tournament_indices = np.random.choice(POPULATION_SIZE, 2, replace=False)
            if fitness_scores[tournament_indices[0]] < fitness_scores[tournament_indices[1]]:
                parent_indices.append(tournament_indices[0])
            else:
                parent_indices.append(tournament_indices[1])

        next_generation = []
        for i in range(POPULATION_SIZE//2):
            parent_1 = population[parent_indices[2*i]]
            parent_2 = population[parent_indices[2*i+1]]

          
            child_1, child_2 = crossover(parent_1, parent_2)

            child_1 = mutate(child_1)
            child_2 = mutate(child_2)

            next_generation.append(child_1)
            next_generation.append(child_2)

        population = np.array(next_generation)

        best_fitness = np.min(fitness_scores)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    
        if best_fitness == 0:
            break

    best_individual = population[np.argmin(calculate_fitness(population, donor_homo, donor_lumo, acceptor_lumo, acceptor_homo))]

    # Best pair of triangles
    displacement_vector = best_individual.reshape((3, 3))
    best_triangle_2 = triangle_2 + displacement_vector
    return triangle_1, best_triangle_2

## shifting function
def shift_triangle(triangle_1, triangle_2, distance, shift_second_triangle=True):
    normal_vector = np.cross(triangle_1[1] - triangle_1[0], triangle_1[2] - triangle_1[0])
    
    if shift_second_triangle:
        displacement_vector = normal_vector / np.linalg.norm(normal_vector) * distance
        shifted_triangle = triangle_2 + displacement_vector
    else:
        displacement_vector = -normal_vector / np.linalg.norm(normal_vector) * distance
        shifted_triangle = triangle_1 + displacement_vector
    
    return shifted_triangle


## PDB file
def WritePDB(file, atoms):
   #  Writes a pdb file for atoms.

    pdb = open(file,"w")
    
    strfrm = "ATOM  {:5d} {:4s} {:4s}{:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}" + \
                "{:6.2f}{:6.2f}          {:2s}\n"
    natm = len(atoms) 
    for iatm in range(1,natm+1):
        atom = atoms[iatm -1]
        pdb.write(strfrm.format(iatm,atom.name,atom.resi,atom.segm[:1],
                                atom.ires,atom.r.x, atom.r.y, atom.r.z,
                                atom.occp,atom.beta,atom.symb))
    pdb.write("END\n")
    pdb.close()



def main():
    triangle_1, triangle_2 = genetic_algorithm()

   
    print("Initial Triangle 1:", triangle_1)
    print("Triangle 2:", triangle_2)
    print("Initial Triangle 2:", initial_triangle_2)

    # distance between molecules
    d = 3.1
    
    new_triangle_2 = shift_triangle(triangle_1, triangle_2, d)

  # LUMO - HOMO gaps
    print("donor_diff")
    print(donor_lumo - donor_homo)
    
    print("acceptor_diff")
    print(acceptor_lumo-acceptor_homo)

    print("charge transfer")
    print((acceptor_lumo-donor_homo)*27.21)

    atoms = []

    # first triangle - blue color
    for i, coord in enumerate(triangle_1):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="C", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="C")
        atoms.append(atom)

   # second triangle - initial state - red color
    for i, coord in enumerate(initial_triangle_2):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="O", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="O")
        atoms.append(atom)

    # second triangle generated by the algorithm and shifted - yellow color
    for i, coord in enumerate(new_triangle_2):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="O", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="O")
        atoms.append(atom)



    # Write new PDB file
    WritePDB("result.pdb", atoms)
    

if __name__ == '__main__':
    main()
