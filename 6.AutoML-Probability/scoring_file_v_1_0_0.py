# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"age": pd.Series(["24"], dtype="int64"), "job": pd.Series(["technician"], dtype="object"), "marital": pd.Series(["single"], dtype="object"), "education": pd.Series(["university.degree"], dtype="object"), "default": pd.Series(["no"], dtype="object"), "housing": pd.Series(["no"], dtype="object"), "loan": pd.Series(["yes"], dtype="object"), "contact": pd.Series(["cellular"], dtype="object"), "month": pd.Series(["jul"], dtype="object"), "duration": pd.Series(["109"], dtype="int64"), "campaign": pd.Series(["3"], dtype="int64"), "pdays": pd.Series(["999"], dtype="int64"), "previous": pd.Series(["0"], dtype="int64"), "poutcome": pd.Series(["nonexistent"], dtype="object"), "emp.var.rate": pd.Series(["1.4"], dtype="float64"), "cons.price.idx": pd.Series(["93.918"], dtype="float64"), "cons.conf.idx": pd.Series(["-42.7"], dtype="float64"), "euribor3m": pd.Series(["4.963"], dtype="float64"), "nr.employed": pd.Series(["5228.1"], dtype="float64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        path = os.path.normpath(model_path)
        path_split = path.split(os.sep)
        log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result_predict = model.predict(data)
        result_class_name = model.y_transformer.classes_
        result_predict_proba = model.predict_proba(data)

        result_with_score = pd.DataFrame(result_predict_proba, columns=result_class_name).to_json(orient='records')
        result = (("{\"result\": \"%s\", \"score:\": %s}" % (result_predict[0],result_with_score)).encode('ascii', 'ignore')).decode("utf-8")

        return result

    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
