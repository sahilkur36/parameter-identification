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
class PSOSettings:
    population_size: int = 30
    generations: int = 50
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive: float = 2.0
    social: float = 2.0
    velocity_ratio: float = 0.1
    seed: int | None = 42


class PSOOptimizer:
    def __init__(self, settings: PSOSettings | None = None) -> None:
        self.settings = settings or PSOSettings()

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
        if settings.population_size < 1:
            raise ValueError("population_size must be at least 1.")
        if settings.generations < 0:
            raise ValueError("generations must be non-negative.")
        rng = np.random.default_rng(settings.seed)
        dim = len(lower_bounds)
        span = upper_bounds - lower_bounds
        population = lower_bounds + span * rng.random((settings.population_size, dim))
        velocity_bounds = span * settings.velocity_ratio
        velocity = -velocity_bounds + 2.0 * velocity_bounds * rng.random((settings.population_size, dim))

        fitness, forces = evaluate_population(population, objective)
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        pbest_forces = [force.copy() for force in forces]
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
            if settings.generations > 1:
                inertia = settings.inertia_start - (
                    (settings.inertia_start - settings.inertia_end) * (generation - 1) / (settings.generations - 1)
                )
            else:
                inertia = settings.inertia_start
            r1 = rng.random((settings.population_size, dim))
            r2 = rng.random((settings.population_size, dim))
            velocity = (
                inertia * velocity
                + settings.cognitive * r1 * (pbest - population)
                + settings.social * r2 * (global_best - population)
            )
            velocity = np.clip(velocity, -velocity_bounds, velocity_bounds)
            population = np.clip(population + velocity, lower_bounds, upper_bounds)

            fitness, forces = evaluate_population(population, objective)
            improved = fitness < pbest_fitness
            pbest[improved] = population[improved]
            pbest_fitness[improved] = fitness[improved]
            for i, is_improved in enumerate(improved):
                if is_improved:
                    pbest_forces[i] = forces[i].copy()

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
