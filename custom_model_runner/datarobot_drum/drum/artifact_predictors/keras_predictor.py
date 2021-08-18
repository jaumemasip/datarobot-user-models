import pandas as pd

from datarobot_drum.drum.common import (
    PythonArtifacts,
    REGRESSION_PRED_COLUMN,
    extra_deps,
    SupportedFrameworks,
    TargetType,
)
from datarobot_drum.drum.exceptions import DrumCommonException
from datarobot_drum.drum.artifact_predictors.artifact_predictor import ArtifactPredictor


class KerasPredictor(ArtifactPredictor):
    def __init__(self):
        super(KerasPredictor, self).__init__(
            SupportedFrameworks.KERAS, PythonArtifacts.KERAS_EXTENSION
        )
        self._model = None

    def is_framework_present(self):
        try:
            from tensorflow.keras.models import load_model

            return True
        except ImportError:
            return False

    def framework_requirements(self):
        return extra_deps[SupportedFrameworks.KERAS]

    def can_load_artifact(self, artifact_path):
        if not self.is_artifact_supported(artifact_path):
            return False

        try:
            from tensorflow.keras.models import load_model

            return True
        except ImportError:
            return False

    def can_use_model(self, model):
        if not self.is_framework_present():
            return False

        try:
            from sklearn.pipeline import Pipeline
            from tensorflow import keras as keras_tf

            if isinstance(model, Pipeline):
                # check the final estimator in the pipeline is Keras
                if isinstance(model[-1], keras_tf.Model):
                    return True
            elif isinstance(model, keras_tf.Model):
                return True
        except Exception as e:
            self._logger.debug("Exception: {}".format(e))
            return False
        return False

    def load_model_from_artifact(self, artifact_path):
        from tensorflow.keras.models import load_model

        self._model = load_model(artifact_path, compile=False)
        return self._model

    def predict(self, data, model, **kwargs):
        super(KerasPredictor, self).predict(data, model, **kwargs)
        predictions = model.predict(data)
        assert False, "figure out how to get labels out of keras"
