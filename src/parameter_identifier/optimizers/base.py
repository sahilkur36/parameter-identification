from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class ObjectiveFunction(Protocol):
    def __call__(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        ...


ProgressCallback = Callable[[int, int, float], None]
StopCallback = Callable[[], bool]


@dataclass(frozen=True)
class HistoryEntry:
    generation: int
    generation_best_params: np.ndarray
    generation_best_fitness: float
    generation_best_force: np.ndarray
    global_best_params: np.ndarray
    global_best_fitness: float
    global_best_force: np.ndarray


@dataclass
class OptimizationHistory:
    parameter_names: list[str]
    entries: list[HistoryEntry] = field(default_factory=list)

    @property
    def generations(self) -> np.ndarray:
        return np.asarray([entry.generation for entry in self.entries], dtype=int)

    @property
    def generation_best_fitness(self) -> np.ndarray:
        return np.asarray([entry.generation_best_fitness for entry in self.entries], dtype=float)

    @property
    def global_best_fitness(self) -> np.ndarray:
        return np.asarray([entry.global_best_fitness for entry in self.entries], dtype=float)

    def parameter_series(self, parameter_index: int) -> tuple[np.ndarray, np.ndarray]:
        generation_best = np.asarray(
            [entry.generation_best_params[parameter_index] for entry in self.entries],
            dtype=float,
        )
        global_best = np.asarray(
            [entry.global_best_params[parameter_index] for entry in self.entries],
            dtype=float,
        )
        return generation_best, global_best


def evaluate_population(population: np.ndarray, objective: ObjectiveFunction) -> tuple[np.ndarray, list[np.ndarray]]:
    fitness = np.zeros(population.shape[0], dtype=float)
    forces: list[np.ndarray] = []
    for i, row in enumerate(population):
        fitness_value, force = objective(row)
        fitness[i] = fitness_value
        forces.append(force)
    return fitness, forces


def make_history_entry(
    generation: int,
    population: np.ndarray,
    fitness: np.ndarray,
    forces: list[np.ndarray],
    global_best_params: np.ndarray,
    global_best_fitness: float,
    global_best_force: np.ndarray,
) -> HistoryEntry:
    best_index = int(np.argmin(fitness))
    return HistoryEntry(
        generation=generation,
        generation_best_params=population[best_index].copy(),
        generation_best_fitness=float(fitness[best_index]),
        generation_best_force=forces[best_index].copy(),
        global_best_params=global_best_params.copy(),
        global_best_fitness=float(global_best_fitness),
        global_best_force=global_best_force.copy(),
    )
