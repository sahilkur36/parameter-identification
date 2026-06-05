from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreprocessSettings:
    skeleton_method: int = 2
    force_scale: float = 1.0


@dataclass(frozen=True)
class PreprocessResult:
    displacement: np.ndarray
    force: np.ndarray
    skeleton_indices: np.ndarray


def load_experiment_file(path: str, force_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Experimental data must contain at least two columns: displacement and force.")
    displacement = np.asarray(data[:, 0], dtype=float)
    force = np.asarray(data[:, 1], dtype=float) * float(force_scale)
    if len(displacement) < 2:
        raise ValueError("Experimental data must contain at least two rows.")
    return displacement, force


def preprocess_curve(
    displacement: np.ndarray,
    force: np.ndarray,
    settings: PreprocessSettings | None = None,
) -> PreprocessResult:
    settings = settings or PreprocessSettings()
    u, f = delete_repetitive_points(displacement, force)
    skeleton_indices = get_skeleton_indices(u, f, settings.skeleton_method)
    return PreprocessResult(
        displacement=np.asarray(u, dtype=float),
        force=np.asarray(f, dtype=float),
        skeleton_indices=np.asarray(skeleton_indices, dtype=int),
    )


def delete_repetitive_points(displacement: np.ndarray, force: np.ndarray) -> tuple[list[float], list[float]]:
    u = np.insert(np.asarray(displacement, dtype=float), 0, 0.0)
    f = np.insert(np.asarray(force, dtype=float), 0, 0.0)
    u_new: list[float] = []
    f_new: list[float] = []
    for i in range(len(u) - 1):
        if u[i] != u[i + 1]:
            u_new.append(float(u[i]))
            f_new.append(float(f[i]))
    u_new.append(float(u[-1]))
    f_new.append(float(f[-1]))
    return u_new, f_new


def get_skeleton_indices(displacement: list[float], force: list[float], skeleton_method: int) -> list[int]:
    u = np.asarray(displacement, dtype=float)
    f = np.asarray(force, dtype=float)
    tag: list[int] = []
    i0 = 0
    for i in range(1, len(u) - 1):
        if (u[i + 1] - u[i]) * (u[i] - u[i - 1]) < 0:
            if skeleton_method == 1:
                tag.append(i)
            elif skeleton_method == 2:
                if i <= i0:
                    continue
                if u[i] > u[i - 1]:
                    tag.append(i0 + int(np.argmax(f[i0:i])))
                else:
                    tag.append(i0 + int(np.argmin(f[i0:i])))
            else:
                raise ValueError("skeleton_method must be 1 or 2.")
            i0 = i
    return tag
