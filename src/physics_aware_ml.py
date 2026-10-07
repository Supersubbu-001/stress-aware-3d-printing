#!/usr/bin/env python3
"""Physics-aware ML model for adaptive infill optimization."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .utils import get_logger

logger = get_logger(__name__)


class PhysicsAwareMLModel:
    """Simple physics-aware surrogate model for mapping stress to infill density."""

    def __init__(self, config):
        self.config = config
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
        stress = np.asarray(stress_values, dtype=float).ravel()
        if stress.size == 0:
            return np.array([], dtype=float)

        if geometry_features is None:
            geom = np.zeros((len(stress), 4))
        else:
            geom = np.asarray(geometry_features, dtype=float)
            if geom.shape[0] != len(stress):
                geom = np.tile(np.array([1.0, 0.5, 0.25, 0.1], dtype=float), (len(stress), 1))

        features = np.column_stack([
            stress,
            np.clip(geom[:, 0], 0, 1),
            np.clip(geom[:, 1], 0, 1),
            np.clip(geom[:, 2], 0, 1),
        ])

        max_stress = np.max(stress) if stress.size else 1.0
        rule_based = 10 + (stress / (max_stress + 1e-8)) * 90
        correction = self.model.predict(features)
        final = 0.7 * rule_based + 0.3 * np.clip(correction, 10, 100)
        return np.clip(final, 10, 100)
