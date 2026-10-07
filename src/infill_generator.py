#!/usr/bin/env python3
"""Physics-aware ML model for adaptive infill optimization with training."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .utils import get_logger

logger = get_logger(__name__)


class PhysicsAwareMLModel:
    """Physics-aware ML model for mapping stress to infill density."""

    def __init__(self, config):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("Physics-aware ML model initialized (untrained)")

    def train(self, stress_values, target_density):
        """Train the ML model on stress data and target density values."""
        try:
            stress = np.asarray(stress_values, dtype=float).ravel()
            density = np.asarray(target_density, dtype=float).ravel()

            if len(stress) != len(density):
                raise ValueError(f"Stress and density arrays must have same length: {len(stress)} vs {len(density)}")

            if len(stress) < 2:
                logger.warning("Insufficient training data (< 2 samples). Using physics-based prediction only.")
                self.is_trained = False
                return

            logger.info(f"Training ML model with {len(stress)} samples...")

            max_stress = np.max(stress) + 1e-8
            stress_norm = stress / max_stress

            features = np.column_stack([
                stress_norm,
                stress_norm ** 2,
                np.sqrt(np.clip(stress_norm, 0, 1)),
                np.log(stress_norm + 1),
            ])

            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, density)
            self.is_trained = True

            train_pred = self.model.predict(features_scaled)
            rmse = np.sqrt(np.mean((train_pred - density) ** 2))
            mae = np.mean(np.abs(train_pred - density))

            logger.info("✅ Model trained successfully!")
            logger.info(f"   Training RMSE: {rmse:.2f}%")
            logger.info(f"   Training MAE: {mae:.2f}%")
            logger.info("   Feature importance:")
            for i, imp in enumerate(self.model.feature_importances_):
                names = ["Stress", "Stress²", "√Stress", "Log(Stress)"]
                logger.info(f"     {names[i]}: {imp:.4f}")

        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.is_trained = False
            raise

    def predict(self, stress_values, geometry_features=None):
        """Predict infill density from stress values."""
        stress = np.asarray(stress_values, dtype=float).ravel()

        if stress.size == 0:
            return np.array([], dtype=float)

        if not self.is_trained:
            logger.warning("Model not trained. Using physics-based prediction only.")
            return self._physics_based(stress, geometry_features)

        try:
            max_stress = np.max(stress) + 1e-8
            stress_norm = stress / max_stress

            features = np.column_stack([
                stress_norm,
                stress_norm ** 2,
                np.sqrt(np.clip(stress_norm, 0, 1)),
                np.log(stress_norm + 1),
            ])

            features_scaled = self.scaler.transform(features)
            ml_density = self.model.predict(features_scaled)
            physics_density = self._physics_based(stress, geometry_features)
            final_density = 0.8 * ml_density + 0.2 * physics_density

            logger.info(f"ML prediction - Density range: {final_density.min():.1f}% - {final_density.max():.1f}%")
            return np.clip(final_density, 10, 100)

        except Exception as e:
            logger.error(f"Error during ML prediction: {e}")
            return self._physics_based(stress, geometry_features)

    def _physics_based(self, stress, geometry_features=None):
        """Fallback physics-only fill rule."""
        stress = np.asarray(stress, dtype=float).ravel()
        max_stress = np.max(stress) if stress.size else 1.0
        min_stress = np.min(stress) if stress.size else 0.0

        if max_stress == min_stress:
            normalized = np.ones_like(stress) * 0.5
        else:
            normalized = (stress - min_stress) / (max_stress - min_stress)

        infill_density = 20 + normalized * 80

        if geometry_features is not None:
            geom = np.asarray(geometry_features, dtype=float)
            if geom.shape[0] == len(stress) and geom.shape[1] > 0:
                curvature = np.clip(geom[:, 0], 0, 1)
                infill_density = infill_density * (0.8 + 0.4 * curvature)

        logger.info(f"Physics-based prediction - Density range: {infill_density.min():.1f}% - {infill_density.max():.1f}%")
        return np.clip(infill_density, 10, 100)
