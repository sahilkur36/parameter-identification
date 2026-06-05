from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parameter_identifier.optimizers.base import (
    ObjectiveFunction,
    OptimizationHistory,
    ProgressCallback,
    StopCallback,
    evaluate_population,
    make_history_entry,
)


@dataclass
class GASettings:
    population_size: int = 30
    generations: int = 50
    crossover_rate: float = 0.85
    mutation_rate: float = 0.12
    mutation_scale: float = 0.08
    tournament_size: int = 3
    elite_count: int = 1
    seed: int | None = 42


class GAOptimizer:
    def __init__(self, settings: GASettings | None = None) -> None:
        self.settings = settings or GASettings()

    def run(
        self,
        objective: ObjectiveFunction,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        parameter_names: list[str],
        progress: ProgressCallback | None = None,
        should_stop: StopCallback | None = None,
    ) -> OptimizationHistory:
        settings = self.settings
        if settings.population_size < 2:
            raise ValueError("GA population_size must be at least 2.")
        if settings.generations < 0:
            raise ValueError("generations must be non-negative.")
        rng = np.random.default_rng(settings.seed)
        dim = len(lower_bounds)
        span = upper_bounds - lower_bounds
        population = lower_bounds + span * rng.random((settings.population_size, dim))

        fitness, forces = evaluate_population(population, objective)
        best_index = int(np.argmin(fitness))
        global_best = population[best_index].copy()
        global_best_fitness = float(fitness[best_index])
        global_best_force = forces[best_index].copy()
        history = OptimizationHistory(parameter_names=parameter_names)
        history.entries.append(
            make_history_entry(0, population, fitness, forces, global_best, global_best_fitness, global_best_force)
        )
        if progress:
            progress(0, settings.generations, global_best_fitness)

        for generation in range(1, settings.generations + 1):
            if should_stop and should_stop():
                break
            elite_count = min(settings.elite_count, settings.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            next_population = [population[index].copy() for index in elite_indices]

            while len(next_population) < settings.population_size:
                parent1 = self._tournament_select(population, fitness, settings.tournament_size, rng)
                parent2 = self._tournament_select(population, fitness, settings.tournament_size, rng)
                child1, child2 = parent1.copy(), parent2.copy()
                if rng.random() < settings.crossover_rate:
                    alpha = rng.random(dim)
                    child1 = alpha * parent1 + (1.0 - alpha) * parent2
                    child2 = alpha * parent2 + (1.0 - alpha) * parent1
                child1 = self._mutate(child1, lower_bounds, upper_bounds, span, settings, rng)
                child2 = self._mutate(child2, lower_bounds, upper_bounds, span, settings, rng)
                next_population.append(child1)
                if len(next_population) < settings.population_size:
                    next_population.append(child2)

            population = np.asarray(next_population, dtype=float)
            fitness, forces = evaluate_population(population, objective)
            best_index = int(np.argmin(fitness))
            if fitness[best_index] < global_best_fitness:
                global_best = population[best_index].copy()
                global_best_fitness = float(fitness[best_index])
                global_best_force = forces[best_index].copy()

            history.entries.append(
                make_history_entry(
                    generation,
                    population,
                    fitness,
                    forces,
                    global_best,
                    global_best_fitness,
                    global_best_force,
                )
            )
            if progress:
                progress(generation, settings.generations, global_best_fitness)
        return history

    @staticmethod
    def _tournament_select(
        population: np.ndarray,
        fitness: np.ndarray,
        tournament_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        size = min(max(1, tournament_size), len(population))
        indices = rng.choice(len(population), size=size, replace=False)
        return population[indices[int(np.argmin(fitness[indices]))]]

    @staticmethod
    def _mutate(
        child: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        span: np.ndarray,
        settings: GASettings,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask = rng.random(len(child)) < settings.mutation_rate
        if np.any(mask):
            child = child.copy()
            child[mask] += rng.normal(0.0, settings.mutation_scale, size=int(np.sum(mask))) * span[mask]
        return np.clip(child, lower_bounds, upper_bounds)
