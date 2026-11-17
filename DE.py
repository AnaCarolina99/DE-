import numpy as np
import pandas as pd
import time


class IIC_IS_DE:
    """
    IIC-IS:
    Differential Evolution para Instance Selection
    baseado no artigo:

    A Differential Evolution Approach to Feature Selection
    and Instance Selection

    Utiliza DE/best/1 + IIC Fitness.
    """

    def __init__(
        self,
        population_size=80,
        generations=100,
        F=0.5,
        alpha=0.4,
        random_state=None
    ):

        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.alpha = alpha
        self.random_state = random_state

        self.population = None
        self.fitness_values = None

        self.best_solution = None
        self.best_binary = None
        self.best_fitness = np.inf

        self.execution_time = 0

    # ==========================================================
    # SIGMOID
    # ==========================================================

    def sigmoid(self, x):

        # Evita overflow numérico
        x = np.clip(x, -500, 500)

        return 1.0 / (1.0 + np.exp(-x))

    # ==========================================================
    # CONVERSÃO CONTÍNUO -> BINÁRIO
    # ==========================================================

    def continuous_to_binary(self, solution):

        probability = self.sigmoid(solution)

        random_values = self.rng.random(len(solution))

        binary_solution = (
            random_values < probability
        ).astype(int)

        return binary_solution

    # ==========================================================
    # INICIALIZAÇÃO DA POPULAÇÃO
    # ==========================================================

    def initialize_population(self, n_instances):

        self.population = self.rng.uniform(
            low=-1,
            high=1,
            size=(
                self.population_size,
                n_instances
            )
        )

    # ==========================================================
    # REPRESENTATIVE SET (ReS)
    # ==========================================================

    def representative_sets(
        self,
        X,
        y,
        selected_indices
    ):

        representative_sets = []

        classes = np.unique(y)

        for class_label in classes:

            # Índices selecionados pertencentes à classe
            class_indices = selected_indices[
                y[selected_indices] == class_label
            ]

            # Se nenhuma instância da classe foi selecionada
            if len(class_indices) == 0:

                return None

            class_data = X[class_indices]

            # --------------------------------------------------
            # Centróide da classe
            # --------------------------------------------------

            centroid = np.mean(
                class_data,
                axis=0
            )

            # --------------------------------------------------
            # Distância de cada instância ao centróide
            # --------------------------------------------------

            distances = np.linalg.norm(
                class_data - centroid,
                axis=1
            )

            # --------------------------------------------------
            # 10% das instâncias + centróide
            # --------------------------------------------------

            number_representatives = max(
                1,
                int(np.ceil(0.10 * len(class_indices)))
            )

            nearest_order = np.argsort(distances)

            nearest_indices = nearest_order[
                :number_representatives
            ]

            representatives = class_data[
                nearest_indices
            ]

            # Adiciona o centróide
            representatives = np.vstack(
                [representatives, centroid]
            )

            representative_sets.append(
                representatives
            )

        return representative_sets

    # ==========================================================
    # ALGORITMO 1
    # INTERCLASS DISTANCE
    # ==========================================================

    def interclass_distance(
        self,
        X,
        y,
        selected_indices
    ):

        representative_sets = self.representative_sets(
            X,
            y,
            selected_indices
        )

        if representative_sets is None:
            return 0.0

        number_classes = len(
            representative_sets
        )

        if number_classes < 2:
            return 0.0

        total_distance = 0.0

        number_class_pairs = 0

        # ------------------------------------------------------
        # Para cada par de classes
        # ------------------------------------------------------

        for a in range(number_classes):

            for b in range(
                a + 1,
                number_classes
            ):

                ReS_a = representative_sets[a]
                ReS_b = representative_sets[b]

                class_distance = 0.0
                links = 0

                # --------------------------------------------------
                # Distância entre todas as instâncias dos
                # dois Representative Sets
                # --------------------------------------------------

                for instance_a in ReS_a:

                    for instance_b in ReS_b:

                        distance = np.linalg.norm(
                            instance_a - instance_b
                        )

                        class_distance += distance
                        links += 1

                if links > 0:

                    average_class_distance = (
                        class_distance / links
                    )

                    total_distance += (
                        average_class_distance
                    )

                    number_class_pairs += 1

        if number_class_pairs == 0:
            return 0.0

        # Média das distâncias entre classes
        distance = (
            total_distance /
            number_class_pairs
        )

        return distance

    # ==========================================================
    # ALGORITMO 2
    # INTRACLASS SPREAD
    # ==========================================================

    def intraclass_spread(
        self,
        X,
        y,
        selected_indices
    ):

        selected_X = X[selected_indices]
        selected_y = y[selected_indices]

        classes = np.unique(y)

        total_spread = 0.0

        # ------------------------------------------------------
        # Para cada feature
        # ------------------------------------------------------

        number_features = X.shape[1]

        for feature in range(
            number_features
        ):

            feature_spread = 0.0

            # --------------------------------------------------
            # Para cada classe
            # --------------------------------------------------

            for class_label in classes:

                class_values = selected_X[
                    selected_y == class_label,
                    feature
                ]

                # Caso a classe tenha somente uma instância
                if len(class_values) <= 1:

                    std = 0.0

                else:

                    std = np.std(
                        class_values
                    )

                feature_spread += std

            total_spread += feature_spread

        return total_spread

    # ==========================================================
    # FITNESS
    # ==========================================================

    def fitness(
        self,
        X,
        y,
        binary_solution
    ):

        selected_indices = np.where(
            binary_solution == 1
        )[0]

        # ------------------------------------------------------
        # Não permite solução vazia
        # ------------------------------------------------------

        if len(selected_indices) == 0:
            return np.inf

        # ------------------------------------------------------
        # Todas as classes precisam estar representadas
        # ------------------------------------------------------

        selected_classes = np.unique(
            y[selected_indices]
        )

        all_classes = np.unique(y)

        if len(selected_classes) != len(all_classes):

            return np.inf

        # ------------------------------------------------------
        # Intraclass Spread
        # ------------------------------------------------------

        spread = self.intraclass_spread(
            X,
            y,
            selected_indices
        )

        # ------------------------------------------------------
        # Interclass Distance
        # ------------------------------------------------------

        distance = self.interclass_distance(
            X,
            y,
            selected_indices
        )

        # Evita divisão por zero
        if distance <= 0:

            return np.inf

        # ------------------------------------------------------
        # Equação 5 do artigo
        #
        # Fitness =
        # (Spread + alpha * |F|) / Distance
        # ------------------------------------------------------

        number_features = X.shape[1]

        fitness_value = (
            spread +
            self.alpha * number_features
        ) / distance

        return fitness_value

    # ==========================================================
    # AVALIAÇÃO DE TODA A POPULAÇÃO
    # ==========================================================

    def evaluate_population(
        self,
        X,
        y
    ):

        fitness_values = []

        binary_population = []

        for individual in self.population:

            binary_solution = (
                self.continuous_to_binary(
                    individual
                )
            )

            value = self.fitness(
                X,
                y,
                binary_solution
            )

            fitness_values.append(value)

            binary_population.append(
                binary_solution
            )

        return (
            np.array(fitness_values),
            binary_population
        )

    # ==========================================================
    # DE/BEST/1
    # ==========================================================

    def mutation_best_1(
        self,
        target_index
    ):

        # ------------------------------------------------------
        # Melhor indivíduo atual
        # ------------------------------------------------------

        best_index = np.argmin(
            self.fitness_values
        )

        x_best = self.population[
            best_index
        ]

        # ------------------------------------------------------
        # Escolhe r1 e r2 diferentes
        # ------------------------------------------------------

        possible_indices = list(
            range(self.population_size)
        )

        possible_indices.remove(
            target_index
        )

        if best_index in possible_indices:

            possible_indices.remove(
                best_index
            )

        r1, r2 = self.rng.choice(
            possible_indices,
            size=2,
            replace=False
        )

        x_r1 = self.population[r1]
        x_r2 = self.population[r2]

        # ------------------------------------------------------
        # DE/best/1
        #
        # v = x_best + F * (x_r1 - x_r2)
        # ------------------------------------------------------

        mutant = (
            x_best +
            self.F * (x_r1 - x_r2)
        )

        return mutant

    # ==========================================================
    # EXECUÇÃO
    # ==========================================================

    def fit(
        self,
        X,
        y
    ):

        self.rng = np.random.default_rng(
            self.random_state
        )

        start_time = time.time()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_instances = X.shape[0]

        # ------------------------------------------------------
        # Inicialização
        # ------------------------------------------------------

        self.initialize_population(
            n_instances
        )

        # ------------------------------------------------------
        # Avaliação inicial
        # ------------------------------------------------------

        (
            self.fitness_values,
            binary_population
        ) = self.evaluate_population(
            X,
            y
        )

        # ------------------------------------------------------
        # Melhor solução inicial
        # ------------------------------------------------------

        best_index = np.argmin(
            self.fitness_values
        )

        self.best_solution = (
            self.population[
                best_index
            ].copy()
        )

        self.best_binary = (
            binary_population[
                best_index
            ].copy()
        )

        self.best_fitness = (
            self.fitness_values[
                best_index
            ]
        )

        # ======================================================
        # EVOLUÇÃO
        # ======================================================

        for generation in range(
            self.generations
        ):

            for i in range(
                self.population_size
            ):

                # --------------------------------------------------
                # Mutação DE/best/1
                # --------------------------------------------------

                mutant = self.mutation_best_1(
                    i
                )

                # --------------------------------------------------
                # Conversão para binário
                # --------------------------------------------------

                trial_binary = (
                    self.continuous_to_binary(
                        mutant
                    )
                )

                # --------------------------------------------------
                # Fitness da nova solução
                # --------------------------------------------------

                trial_fitness = self.fitness(
                    X,
                    y,
                    trial_binary
                )

                # --------------------------------------------------
                # Seleção
                #
                # Minimização:
                # menor fitness = melhor
                # --------------------------------------------------

                if (
                    trial_fitness
                    <
                    self.fitness_values[i]
                ):

                    self.population[i] = mutant

                    self.fitness_values[i] = (
                        trial_fitness
                    )

                    # --------------------------------------------------
                    # Atualiza melhor solução global
                    # --------------------------------------------------

                    if (
                        trial_fitness
                        <
                        self.best_fitness
                    ):

                        self.best_solution = (
                            mutant.copy()
                        )

                        self.best_binary = (
                            trial_binary.copy()
                        )

                        self.best_fitness = (
                            trial_fitness
                        )

            # --------------------------------------------------
            # Mostra progresso
            # --------------------------------------------------

            selected = np.sum(
                self.best_binary
            )

            print(
                f"Geração "
                f"{generation + 1:3d}/"
                f"{self.generations} | "
                f"Fitness: "
                f"{self.best_fitness:.6f} | "
                f"Instâncias: "
                f"{selected}/{n_instances}"
            )

        self.execution_time = (
            time.time() - start_time
        )

        return self

    # ==========================================================
    # ÍNDICES SELECIONADOS
    # ==========================================================

    def get_selected_indices(self):

        if self.best_binary is None:

            raise RuntimeError(
                "Execute fit() primeiro."
            )

        return np.where(
            self.best_binary == 1
        )[0]

    # ==========================================================
    # DATASET REDUZIDO
    # ==========================================================

    def transform(
        self,
        X,
        y
    ):

        indices = (
            self.get_selected_indices()
        )

        return (
            X[indices],
            y[indices]
        )