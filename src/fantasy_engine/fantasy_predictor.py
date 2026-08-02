# fantasy_predictor.py

import os
import logging

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


##############################################################
# FANTASY PREDICTOR (classifier-only)
#
# app.py instantiates this as FantasyPredictor(FANTASY_CLASSIFIER_PATH)
# — a single path to fantasy_classifier.pkl. There is no regression
# model in this pipeline; classifier_encoders.pkl and
# classifier_feature_columns.pkl are auto-discovered next to it.
##############################################################

class FantasyPredictor:

    def __init__(

        self,

        classifier_path,

        classifier_encoders_path=None,

        classifier_feature_columns_path=None

    ):

        self.classifier = joblib.load(classifier_path)

        model_dir = os.path.dirname(os.path.abspath(classifier_path))

        self.classifier_encoders = self._safe_load(

            classifier_encoders_path or os.path.join(model_dir, "classifier_encoders.pkl")

        )

        self.classifier_feature_columns = self._safe_load(

            classifier_feature_columns_path or os.path.join(model_dir, "classifier_feature_columns.pkl")

        )

        if not self.classifier_encoders:

            logger.warning(

                "classifier_encoders.pkl did not load — categorical columns "
                "(venue/batting_team/bowling_team/phase/player_role/"
                "player_sub_role) will be passed to the classifier as raw "
                "strings and WILL raise an XGBoost dtype error. Check "
                "classifier_encoders_path."

            )

        if not self.classifier_feature_columns:

            logger.warning(

                "classifier_feature_columns.pkl did not load — falling back "
                "to numeric-only column selection, which will silently drop "
                "any categorical feature the model was trained on."

            )


    ##########################################################
    # SAFE ARTIFACT LOADING
    ##########################################################

    def _safe_load(self, path):

        try:

            if path and os.path.exists(path):

                return joblib.load(path)

        except Exception as exc:

            logger.warning("Could not load artifact at %s: %s", path, exc)

        return None


    ##########################################################
    # COLUMN ALIGNMENT
    ##########################################################

    def _select_columns(self, dataframe, columns):

        if not columns:

            return dataframe.select_dtypes(include="number")

        aligned = pd.DataFrame(index=dataframe.index)

        for column in columns:

            if column in dataframe.columns:

                aligned[column] = dataframe[column]

            else:

                aligned[column] = 0.0

        return aligned


    def _apply_encoders(self, dataframe):

        if not self.classifier_encoders:

            return dataframe

        dataframe = dataframe.copy()

        encoded_columns = set()

        for column, encoder in self.classifier_encoders.items():

            if column not in dataframe.columns:

                continue

            try:

                known = set(getattr(encoder, "classes_", []))

                dataframe[column] = dataframe[column].apply(

                    lambda value: value if value in known else (encoder.classes_[0] if len(known) else value)

                )

                dataframe[column] = encoder.transform(dataframe[column])

                encoded_columns.add(column)

            except Exception as exc:

                logger.error("Failed to apply encoder for column %s: %s", column, exc)

                raise

        # Any column the classifier expects that's still non-numeric after
        # encoding (i.e. no entry in classifier_encoders for it) is exactly
        # what produces the "DataFrame.dtypes ... object" XGBoost error —
        # surface it clearly here instead of letting XGBoost raise a
        # generic message with no indication of WHICH artifact is stale.
        leftover_object_columns = [

            column for column in dataframe.columns

            if column not in encoded_columns and dataframe[column].dtype == object

        ]

        if leftover_object_columns:

            raise ValueError(

                f"Columns {leftover_object_columns} are still non-numeric after "
                "encoding — classifier_encoders.pkl has no entry for them. "
                "Re-save classifier_encoders.pkl from the training notebook "
                "(it must contain an encoder for every column in "
                "CATEGORICAL_COLS: venue, batting_team, bowling_team, phase, "
                "player_role, player_sub_role)."

            )

        return dataframe


    ##########################################################
    # CLASSIFIER SCORING
    ##########################################################

    def _score_classifier(self, dataframe):

        features = self._select_columns(dataframe, self.classifier_feature_columns)
        features = self._apply_encoders(features)

        if hasattr(self.classifier, "predict_proba"):

            probabilities = self.classifier.predict_proba(features)

            # Assume the positive ("high performer") class is the last column.
            return probabilities[:, -1]

        return self.classifier.predict(features)


    ##########################################################
    # SINGLE PLAYER
    ##########################################################

    def predict_player(

        self,

        features

    ):

        df = pd.DataFrame([features])

        return self.predict_players(df).iloc[0]


    ##########################################################
    # MULTIPLE PLAYERS
    ##########################################################

    def predict_players(

        self,

        dataframe

    ):

        dataframe = dataframe.copy()

        dataframe["high_performer_probability"] = self._score_classifier(dataframe)

        return dataframe


    ##########################################################
    # PROBABILITIES (raw classifier output, all classes)
    ##########################################################

    def predict_probabilities(

        self,

        dataframe

    ):

        if not hasattr(self.classifier, "predict_proba"):

            return None

        features = self._select_columns(dataframe, self.classifier_feature_columns)
        features = self._apply_encoders(features)

        return self.classifier.predict_proba(features)


    ##########################################################
    # RANKED PLAYERS BY HIGH-PERFORMER PROBABILITY
    ##########################################################

    def rank_players(

        self,

        dataframe

    ):

        predictions = self.predict_players(dataframe)

        predictions = predictions.sort_values(

            "high_performer_probability",

            ascending=False

        )

        predictions = predictions.reset_index(drop=True)

        return predictions