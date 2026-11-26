import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


# ============================================================
# Application and Model Configuration
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "superkart_model.joblib"

MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

NUMERICAL_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]

ALLOWED_CATEGORIES = {
    "Product_Sugar_Content": {
        "Low Sugar",
        "No Sugar",
        "Regular",
    },
    "Store_Size": {
        "High",
        "Medium",
        "Small",
    },
    "Store_Location_City_Type": {
        "Tier 1",
        "Tier 2",
        "Tier 3",
    },
    "Store_Type": {
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    },
    "Product_Id_char": {
        "DR",
        "FD",
        "NC",
    },
    "Product_Type_Category": {
        "Non Perishables",
        "Perishables",
    },
}

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model artifact was not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# Request Validation Helpers
# ============================================================

class RequestValidationError(ValueError):
    """Raised when an inference request fails validation."""


def validate_record(record):
    """Validate one record and return a normalized feature dictionary."""

    if not isinstance(record, dict):
        raise RequestValidationError(
            "Each prediction record must be a JSON object."
        )

    received_features = set(record.keys())
    required_features = set(MODEL_FEATURES)

    missing_features = sorted(
        required_features - received_features
    )

    unexpected_features = sorted(
        received_features - required_features
    )

    if missing_features:
        raise RequestValidationError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    if unexpected_features:
        raise RequestValidationError(
            "Unexpected features: "
            + ", ".join(unexpected_features)
        )

    validated_record = {}

    for feature_name in NUMERICAL_FEATURES:
        raw_value = record[feature_name]

        if isinstance(raw_value, bool):
            raise RequestValidationError(
                f"{feature_name} must be a finite number."
            )

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            raise RequestValidationError(
                f"{feature_name} must be a finite number."
            )

        if not math.isfinite(numeric_value):
            raise RequestValidationError(
                f"{feature_name} must be a finite number."
            )

        validated_record[feature_name] = numeric_value

    if validated_record["Product_Weight"] <= 0:
        raise RequestValidationError(
            "Product_Weight must be greater than zero."
        )

    if not (
        0
        <= validated_record["Product_Allocated_Area"]
        <= 1
    ):
        raise RequestValidationError(
            "Product_Allocated_Area must be between 0 and 1."
        )

    if validated_record["Product_MRP"] <= 0:
        raise RequestValidationError(
            "Product_MRP must be greater than zero."
        )

    store_age_value = validated_record[
        "Store_Age_Years"
    ]

    if (
        store_age_value < 0
        or not store_age_value.is_integer()
    ):
        raise RequestValidationError(
            "Store_Age_Years must be a nonnegative integer."
        )

    validated_record["Store_Age_Years"] = int(
        store_age_value
    )

    for feature_name in CATEGORICAL_FEATURES:
        raw_value = record[feature_name]

        if not isinstance(raw_value, str):
            raise RequestValidationError(
                f"{feature_name} must be a text value."
            )

        category_value = raw_value.strip()

        if category_value not in ALLOWED_CATEGORIES[
            feature_name
        ]:
            allowed_values = sorted(
                ALLOWED_CATEGORIES[feature_name]
            )

            raise RequestValidationError(
                f"Invalid {feature_name}. "
                f"Allowed values: {allowed_values}"
            )

        validated_record[feature_name] = (
            category_value
        )

    return {
        feature_name: validated_record[feature_name]
        for feature_name in MODEL_FEATURES
    }


def predict_records(records):
    """Validate records and return finite model predictions."""

    validated_records = [
        validate_record(record)
        for record in records
    ]

    model_input = pd.DataFrame(
        validated_records,
        columns=MODEL_FEATURES,
    )

    predictions = model.predict(model_input)

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "The model produced a nonfinite prediction."
        )

    return [
        round(float(prediction), 2)
        for prediction in predictions
    ]


# ============================================================
# Flask Routes
# ============================================================

@app.get("/")
def index():
    """Return basic backend service information."""

    return jsonify(
        {
            "service": "SuperKart Sales Forecasting API",
            "status": "ready",
            "model_file": MODEL_PATH.name,
            "online_endpoint": "/v1/predict",
            "batch_endpoint": "/v1/predictbatch",
        }
    ), 200


@app.get("/health")
def health():
    """Return backend and model readiness information."""

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "required_feature_count": len(
                MODEL_FEATURES
            ),
        }
    ), 200


@app.post("/v1/predict")
def predict_online():
    """Generate one prediction from one JSON feature record."""

    if not request.is_json:
        return jsonify(
            {
                "error": (
                    "Request Content-Type must be "
                    "application/json."
                )
            }
        ), 415

    payload = request.get_json(
        silent=True
    )

    if payload is None:
        return jsonify(
            {
                "error": "The request body is not valid JSON."
            }
        ), 400

    try:
        predictions = predict_records(
            [payload]
        )

        return jsonify(
            {
                "prediction": predictions[0],
                "prediction_field": (
                    "Product_Store_Sales_Total"
                ),
                "record_count": 1,
                "model": (
                    "Tuned Random Forest"
                ),
                "unseen_category_note": (
                    "Supermarket Type3 is accepted "
                    "technically but was absent from "
                    "historical training data."
                ),
            }
        ), 200

    except RequestValidationError as error:
        return jsonify(
            {
                "error": str(error)
            }
        ), 400

    except Exception:
        app.logger.exception(
            "Online prediction failed."
        )

        return jsonify(
            {
                "error": (
                    "Prediction could not be completed."
                )
            }
        ), 500


@app.post("/v1/predictbatch")
def predict_batch():
    """Generate predictions from multipart CSV or JSON records."""

    try:
        request_format = None

        # ----------------------------------------------------
        # A. Course-provided multipart CSV file contract
        # ----------------------------------------------------

        if "file" in request.files:
            uploaded_file = request.files["file"]

            if (
                uploaded_file is None
                or uploaded_file.filename == ""
            ):
                raise RequestValidationError(
                    "A nonempty CSV file must be "
                    "submitted using the file field."
                )

            try:
                batch_dataframe = pd.read_csv(
                    uploaded_file
                )

            except Exception as error:
                raise RequestValidationError(
                    "The uploaded file could not "
                    "be read as CSV."
                ) from error

            if batch_dataframe.empty:
                raise RequestValidationError(
                    "The uploaded CSV file "
                    "contains no records."
                )

            missing_columns = [
                feature_name
                for feature_name in MODEL_FEATURES
                if feature_name
                not in batch_dataframe.columns
            ]

            unexpected_columns = [
                column_name
                for column_name
                in batch_dataframe.columns
                if column_name
                not in MODEL_FEATURES
            ]

            if missing_columns:
                raise RequestValidationError(
                    "Missing required columns: "
                    + ", ".join(
                        missing_columns
                    )
                )

            if unexpected_columns:
                raise RequestValidationError(
                    "Unexpected columns: "
                    + ", ".join(
                        unexpected_columns
                    )
                )

            batch_dataframe = (
                batch_dataframe.loc[
                    :,
                    MODEL_FEATURES,
                ]
                .copy()
            )

            if (
                batch_dataframe
                .isna()
                .any()
                .any()
            ):
                raise RequestValidationError(
                    "The uploaded CSV contains "
                    "missing feature values."
                )

            records = (
                batch_dataframe
                .to_dict(
                    orient="records"
                )
            )

            request_format = (
                "multipart_csv"
            )

        # ----------------------------------------------------
        # B. Existing JSON records contract
        # ----------------------------------------------------

        elif request.is_json:
            payload = request.get_json(
                silent=True
            )

            if payload is None:
                raise RequestValidationError(
                    "The request body is not "
                    "valid JSON."
                )

            if isinstance(
                payload,
                dict,
            ):
                records = payload.get(
                    "records"
                )

            else:
                records = payload

            if not isinstance(
                records,
                list,
            ):
                raise RequestValidationError(
                    "Batch request must be a "
                    "JSON list or an object "
                    "containing a records list."
                )

            request_format = (
                "json_records"
            )

        else:
            return jsonify(
                {
                    "error": (
                        "Submit either a multipart "
                        "CSV file using the file "
                        "field or an application/json "
                        "batch request."
                    )
                }
            ), 415

        if len(records) == 0:
            raise RequestValidationError(
                "The batch request must contain "
                "at least one record."
            )

        predictions = predict_records(
            records
        )

        prediction_records = [
            {
                "record_number": (
                    record_number
                ),
                "prediction": (
                    prediction_value
                ),
            }
            for (
                record_number,
                prediction_value,
            ) in enumerate(
                predictions,
                start=1,
            )
        ]

        return jsonify(
            {
                "model": (
                    "Tuned Random Forest"
                ),
                "prediction_field": (
                    "Product_Store_Sales_Total"
                ),
                "predictions": (
                    prediction_records
                ),
                "record_count": (
                    len(
                        prediction_records
                    )
                ),
                "request_format": (
                    request_format
                ),
                "unseen_category_note": (
                    "Supermarket Type3 is accepted "
                    "technically but was absent from "
                    "historical training data."
                ),
            }
        ), 200

    except RequestValidationError as error:
        return jsonify(
            {
                "error": str(
                    error
                )
            }
        ), 400

    except Exception:
        app.logger.exception(
            "Batch prediction failed."
        )

        return jsonify(
            {
                "error": (
                    "Batch prediction could "
                    "not be completed."
                )
            }
        ), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
    )
