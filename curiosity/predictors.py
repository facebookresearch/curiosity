#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.models import Model

from curiosity.db import verify_checksum, create_sql, Fact


@Predictor.register('curiosity_predictor')
class CuriosityPredictor(Predictor):

    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True) -> None:
        if frozen:
            model.eval()
        self._model = model
        self._dataset_reader = dataset_reader
        self.cuda_device = next(self._model.named_parameters())[1].get_device()

        # Hard coded fact loading
        db_path = os.path.join('dialog_data', 'wiki_sql.sqlite.db')
        engine, session = create_sql(db_path)
        facts = (
            session
            .query(Fact)
            .all()
        )
        self._dataset_reader._fact_lookup = {f.id: f for f in facts}

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:

        dialogs = inputs['dialogs']
        out = []

        for i, d in enumerate(dialogs):
            if i == 30:
                # Early termination to save time
                break

            instance = self._dataset_reader.text_to_instance(d)
            prediction = self.predict_instance(instance)

            # Label predictions for this dialog
            label_prediction = {
                'dialog_id': d['dialog_id']
            }

            for k, v in prediction.items():
                if k != 'loss':
                    label_prediction[k] = np.argmax(v, axis=1).tolist()

            out.append(label_prediction)

        return out
