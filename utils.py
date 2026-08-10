from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation
import numpy as np
from random import random
import networkx as nx

def uniform_cell_selector(vec):
    return list(range(vec.size()))


class IntVectorUniformMutation(VectorNPointMutation):
    """
    Uniform N Point Integer Mutation
    """

    def __init__(self, probability=0.5, arity=1, events=None, probability_for_each=0.1):
        self.probability_for_each = probability_for_each
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events, cell_selector=uniform_cell_selector)
class GraphColoringEvaluator(SimpleIndividualEvaluator):
    def __init__(self, G: nx.Graph, fitness_dict=None, penalty=1000):
        super().__init__()
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.fitness_dict = {} if fitness_dict is None else {}
        self.edge_array = np.array(G.edges(), dtype=int)
        self.penalty = penalty

    def evaluate_individual(self, individual):
        return self.get_graph_coloring_fitness(np.array(individual.vector))

    def get_graph_coloring_fitness(self, colors):
        # key = tuple(colors)
        # if key in self.fitness_dict:
        #     return self.fitness_dict[key]

        color_u = colors[self.edge_array[:, 0]]
        color_v = colors[self.edge_array[:, 1]]
        conflicts = np.sum(color_u == color_v)
        num_colors = len(np.unique(colors))

        if conflicts == 0:
            fitness = (self.n_nodes - num_colors) / (self.n_nodes - 1)
        else:
            fitness = -conflicts * self.penalty - num_colors

        # self.fitness_dict[key] = fitness
        return fitness
