import numpy as np
import random
from scipy.spatial.distance import euclidean


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
    import re

    # read the input file
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
        # return the first triangle found and its vertex indices
        triangle = np.array(triangles[0])
        vertex_indices = [coords.index(list(triangle[i])) for i in range(3)]
        return triangle, vertex_indices
    else:
        return None, None



    # Define the vertices of the two triangles
triangle_1, TRIANGLE_1_INDICES = molCoordHex('molA.txt')
triangle_2, TRIANGLE_2_INDICES = molCoordHex('btt.txt')

    # Genetic algorithm parameters
POPULATION_SIZE = 1000  # Number of individuals in the population
MUTATION_RATE = 0.01  # Probability of a gene mutating
GENERATIONS = 1200  # Number of generations

def genetic_algorithm():
    def calculate_fitness(population):
        fitness_scores = []
        for individual in population:
            # Calculate the new position of triangle_2 by adding the displacement vector
            # to each vertex of triangle_2
            displacement_vector = individual.reshape((3, 3))
            new_triangle_2 = triangle_2 + displacement_vector

            # Calculate the sum of the squared distances between the vertices of the two triangles
            distances = np.sum((triangle_1 - new_triangle_2)**2)

            # Calculate the penalty term for non-parallelism
            normal_1 = np.cross(triangle_1[1] - triangle_1[0], triangle_1[2] - triangle_1[0])
            normal_2 = np.cross(new_triangle_2[1] - new_triangle_2[0], new_triangle_2[2] - new_triangle_2[0])
            parallel_penalty = abs(np.dot(normal_1, normal_2) / (np.linalg.norm(normal_1) * np.linalg.norm(normal_2)) - 1)

            # Add the penalty term to the fitness score
            fitness_scores.append(distances + parallel_penalty)

        return np.array(fitness_scores)



    def crossover(parent_1, parent_2):
       # single-point crossover on two parents to generate two children.
            # Choose a random crossover point
        crossover_point = np.random.randint(1, len(parent_1))

            # Combine the genes of the parents to create the children
        child_1 = np.concatenate([parent_1[:crossover_point], parent_2[crossover_point:]])
        child_2 = np.concatenate([parent_2[:crossover_point], parent_1[crossover_point:]])

        return child_1, child_2

    def mutate(individual):
         # mutation of an individual by randomly changing one gene.
        if np.random.random() < MUTATION_RATE:
            # Choose a random gene to mutate
            gene_index = np.random.randint(len(individual))

            # Check if the gene index is within bounds
            if gene_index*3 + 3 > len(individual):
                return individual
            # Generate a random displacement vector
            displacement = np.random.randn(3)

            # Reshape the displacement vector to match the shape of the gene
            displacement = displacement.reshape((3,))

            # Replace the gene with the new displacement vector
            individual[gene_index*3 : gene_index*3 + 3] = displacement

        return individual


    # Create the initial population
    population = np.random.randn(POPULATION_SIZE, 9)

    # Run the genetic algorithm for the specified number of generations
    for generation in range(GENERATIONS):
        # Calculate the fitness scores for the population
        fitness_scores = calculate_fitness(population)

        # Select the parents for the next generation using tournament selection
        parent_indices = []
        for i in range(POPULATION_SIZE):
            tournament_indices = np.random.choice(POPULATION_SIZE, 2, replace=False)
            if fitness_scores[tournament_indices[0]] < fitness_scores[tournament_indices[1]]:
                parent_indices.append(tournament_indices[0])
            else:
                parent_indices.append(tournament_indices[1])

        # Create the next generation by performing crossover and mutation on the parents
        next_generation = []
        for i in range(POPULATION_SIZE//2):
            parent_1 = population[parent_indices[2*i]]
            parent_2 = population[parent_indices[2*i+1]]

            # Perform crossover to create two children
            child_1, child_2 = crossover(parent_1, parent_2)

            # Mutate the children
            child_1 = mutate(child_1)
            child_2 = mutate(child_2)

            # Add the children to the next generation
            next_generation.append(child_1)
            next_generation.append(child_2)

        # Replace the current population with the next generation
        population = np.array(next_generation)

        # Print the best fitness score of the current generation
        best_fitness = np.min(fitness_scores)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    


        # Check if the termination condition is met (fitness score of 0)
        if best_fitness == 0:
            break

    best_individual = population[np.argmin(calculate_fitness(population))]
    print(f"The coordinates of the best individual are: {best_individual.reshape((3,3))}")

    best_individual = population[np.argmin(calculate_fitness(population))]
    print(f"The coordinates of the best individual are: {best_individual.reshape((3,3))}")

    # Find the best pair of triangles
    displacement_vector = best_individual.reshape((3, 3))
    best_triangle_2 = triangle_2 + displacement_vector
    return triangle_1, best_triangle_2




def shift_triangle(triangle_1, triangle_2, distance, shift_second_triangle=True):
   # shifting one of the triangles from the other with a displacement 
    # Compute the normal vector of the plane defined by the two triangles
    normal_vector = np.cross(triangle_1[1] - triangle_1[0], triangle_1[2] - triangle_1[0])
    
    # Compute the displacement vector as a scalar multiple of the normal vector
    if shift_second_triangle:
        displacement_vector = normal_vector / np.linalg.norm(normal_vector) * distance
        shifted_triangle = triangle_2 + displacement_vector
    else:
        displacement_vector = -normal_vector / np.linalg.norm(normal_vector) * distance
        shifted_triangle = triangle_1 + displacement_vector
    
    return shifted_triangle


## PDB file
def WritePDB(file, atoms):
    #------------------------------------------------------------------------------
    #  Writes a pdb file for atoms.
    #------------------------------------------------------------------------------
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

   
    print("Triangle 1:", triangle_1)
    print("Triangle 2:", triangle_2)

    d = 3.1
    
    new_triangle_2 = shift_triangle(triangle_1, triangle_2, d)

    atom_indices_molA = TRIANGLE_1_INDICES
    atom_indices_btt = TRIANGLE_2_INDICES

  
    print(atom_indices_molA)



   # Create atom objects with updated coordinates
    atoms = []

    # first triangle - blue color
    for i, coord in enumerate(triangle_1):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="C", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="C")
        atoms.append(atom)
    
    # second triangle generated by the algorithm and shifted - yellow color
    for i, coord in enumerate(new_triangle_2):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="S", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="S")
        atoms.append(atom)
     
   # second triangle generated by the algorithm - red color
    for i, coord in enumerate(triangle_2):
        r = VecR3(coord[0], coord[1], coord[2])
        atom = Atom(name="O", resi="1", segm=" ", ires=i+1, r=r, occp=1.00, beta=0.00, symb="O")
        atoms.append(atom)

    # Write new PDB file
    WritePDB("result.pdb", atoms)
    

if __name__ == '__main__':
    main()
