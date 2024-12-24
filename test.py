from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorFlipMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.before_after_publisher import AFTER_OPERATOR_EVENT_NAME, BEFORE_OPERATOR_EVENT_NAME
from eckity.examples.vectorga.knapsack.knapsack_evaluator import KnapsackEvaluator, NUM_ITEMS



def main():
    cross_op = VectorKPointsCrossover(probability=0.5, k=2)
    algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=NUM_ITEMS),
                      population_size=50,
                      # user-defined fitness evaluation method
                      evaluator=KnapsackEvaluator(),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          cross_op,
                          BitStringVectorFlipMutation(probability=0.05)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=500,
        statistics=BestAverageWorstStatistics()
    )
    algo.register(event=BEFORE_OPERATOR_EVENT_NAME, callback=lambda: "after_gen")
    algo.register(event=AFTER_OPERATOR_EVENT_NAME, callback=lambda: "After")
    algo.evolve()



if __name__ == "__main__":
    main()