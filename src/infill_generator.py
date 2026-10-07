"""
Physics-aware ML model abstraction for adaptive infill optimization
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .utils import get_logger

logger = get_logger(__name__)


class PhysicsAwareMLModel:
    """Simple physics-aware surrogate model for mapping stress to infill density."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self._build_model()

    def _build_model(self):
        # A lightweight deterministic surrogate approximates a physics-aware rule based model.
        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=8,
        )
        logger.info("Physics-aware surrogate model initialized")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, stress_values, geometry_features=None):
        stress = np.asarray(stress_values, dtype=float)
        if stress.ndim == 0:
            stress = np.array([float(stress)])

        if geometry_features is None:
            geometry_features = np.zeros((len(stress), 4))
        geom = np.asarray(geometry_features, dtype=float)
        if geom.shape[0] != len(stress):
            geom = np.tile(np.array([1.0, 0.5, 0.25, 0.1], dtype=float), (len(stress), 1))

        features = np.column_stack([
            stress,
            np.clip(geom[:, 0], 0, 1),
            np.clip(geom[:, 1], 0, 1),
            np.clip(geom[:, 2], 0, 1),
        ])

        # Heuristic blend: physics-based stress rules with learned forest correction
        rule_based = 10 + stress / (np.max(stress) + 1e-6) * 90
        if self.model is not None:
            correction = self.model.predict(features)
            final = 0.7 * rule_based + 0.3 * np.clip(correction, 10, 100)
        else:
            final = rule_based

        return np.clip(final, 10, 100)
