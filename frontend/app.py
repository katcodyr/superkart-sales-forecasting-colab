import os

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# Application and Backend Configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
)

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://superkart-backend-container:7860",
).rstrip("/")

ONLINE_ENDPOINT = (
    f"{BACKEND_URL}/v1/predict"
)

BATCH_ENDPOINT = (
    f"{BACKEND_URL}/v1/predictbatch"
)

REQUEST_TIMEOUT_SECONDS = 120

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

CATEGORY_OPTIONS = {
    "Product_Sugar_Content": [
        "Low Sugar",
        "No Sugar",
        "Regular",
    ],
    "Store_Size": [
        "High",
        "Medium",
        "Small",
    ],
    "Store_Location_City_Type": [
        "Tier 1",
        "Tier 2",
        "Tier 3",
    ],
    "Store_Type": [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    ],
    "Product_Id_char": [
        "DR",
        "FD",
        "NC",
    ],
    "Product_Type_Category": [
        "Non Perishables",
        "Perishables",
    ],
}


# ============================================================
# Response and Batch Validation Helpers
# ============================================================

def parse_response(response):
    """Return valid JSON or raise a clear interface error."""

    try:
        response_json = response.json()

    except ValueError as error:
        raise RuntimeError(
            "Backend returned a non-JSON response "
            f"with status {response.status_code}."
        ) from error

    if response.status_code != 200:
        error_message = response_json.get(
            "error",
            "Unknown backend error.",
        )

        raise RuntimeError(
            "Backend request failed with status "
            f"{response.status_code}: "
            f"{error_message}"
        )

    return response_json


def validate_batch_dataframe(
    batch_data,
):
    """Validate an uploaded batch file before calling the API."""

    if batch_data.empty:
        raise ValueError(
            "The uploaded CSV file "
            "contains no records."
        )

    missing_columns = [
        column
        for column in MODEL_FEATURES
        if column not in batch_data.columns
    ]

    unexpected_columns = [
        column
        for column in batch_data.columns
        if column not in MODEL_FEATURES
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    if unexpected_columns:
        raise ValueError(
            "Unexpected columns: "
            + ", ".join(
                unexpected_columns
            )
        )

    if (
        batch_data[
            MODEL_FEATURES
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "The uploaded CSV contains "
            "missing feature values."
        )

    return (
        batch_data[
            MODEL_FEATURES
        ]
        .copy()
    )


# ============================================================
# Page Header
# ============================================================

st.title(
    "SuperKart Sales Forecasting"
)

st.caption(
    "Tuned Random Forest model served "
    "through the Flask prediction API"
)

st.info(
    "Supermarket Type3 is accepted technically, "
    "but it was absent from the historical training "
    "data. Interpret Type3 predictions cautiously."
)

online_tab, batch_tab = st.tabs(
    [
        "Online Prediction",
        "Batch Prediction",
    ]
)


# ============================================================
# Online Prediction Workflow
# ============================================================

with online_tab:
    st.subheader(
        "Enter one product-store record"
    )

    with st.form(
        "online_prediction_form"
    ):
        (
            left_column,
            right_column,
        ) = st.columns(2)

        with left_column:
            product_weight = (
                st.number_input(
                    "Product Weight",
                    min_value=0.01,
                    value=12.66,
                    step=0.01,
                )
            )

            product_sugar_content = (
                st.selectbox(
                    "Product Sugar Content",
                    CATEGORY_OPTIONS[
                        "Product_Sugar_Content"
                    ],
                )
            )

            product_allocated_area = (
                st.number_input(
                    "Product Allocated Area",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.027,
                    step=0.001,
                    format="%.3f",
                )
            )

            product_mrp = (
                st.number_input(
                    "Product MRP",
                    min_value=0.01,
                    value=117.08,
                    step=0.01,
                )
            )

            store_size = (
                st.selectbox(
                    "Store Size",
                    CATEGORY_OPTIONS[
                        "Store_Size"
                    ],
                    index=1,
                )
            )

        with right_column:
            store_location_city_type = (
                st.selectbox(
                    "Store Location City Type",
                    CATEGORY_OPTIONS[
                        "Store_Location_City_Type"
                    ],
                    index=1,
                )
            )

            store_type = (
                st.selectbox(
                    "Store Type",
                    CATEGORY_OPTIONS[
                        "Store_Type"
                    ],
                    index=3,
                )
            )

            product_id_char = (
                st.selectbox(
                    "Product ID Prefix",
                    CATEGORY_OPTIONS[
                        "Product_Id_char"
                    ],
                    index=1,
                )
            )

            store_age_years = (
                st.number_input(
                    "Store Age in Years",
                    min_value=0,
                    value=16,
                    step=1,
                )
            )

            product_type_category = (
                st.selectbox(
                    "Product Type Category",
                    CATEGORY_OPTIONS[
                        "Product_Type_Category"
                    ],
                )
            )

        submit_online = (
            st.form_submit_button(
                "Predict Sales",
                type="primary",
                use_container_width=True,
            )
        )

    if submit_online:
        online_payload = {
            "Product_Weight": float(
                product_weight
            ),
            "Product_Sugar_Content": (
                product_sugar_content
            ),
            "Product_Allocated_Area": float(
                product_allocated_area
            ),
            "Product_MRP": float(
                product_mrp
            ),
            "Store_Size": (
                store_size
            ),
            "Store_Location_City_Type": (
                store_location_city_type
            ),
            "Store_Type": (
                store_type
            ),
            "Product_Id_char": (
                product_id_char
            ),
            "Store_Age_Years": int(
                store_age_years
            ),
            "Product_Type_Category": (
                product_type_category
            ),
        }

        try:
            online_response = requests.post(
                ONLINE_ENDPOINT,
                json=online_payload,
                timeout=(
                    REQUEST_TIMEOUT_SECONDS
                ),
            )

            online_result = parse_response(
                online_response
            )

            prediction_value = float(
                online_result[
                    "prediction"
                ]
            )

            if not np.isfinite(
                prediction_value
            ):
                raise RuntimeError(
                    "The backend returned a "
                    "nonfinite prediction."
                )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted Product-Store "
                "Sales Total",
                f"{prediction_value:,.2f}",
            )

            st.json(
                online_result
            )

        except (
            requests.RequestException
        ) as error:
            st.error(
                "Could not connect to "
                "the backend: "
                f"{error}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            st.error(
                str(error)
            )


# ============================================================
# Batch Prediction Workflow
# ============================================================

with batch_tab:
    st.subheader(
        "Upload a ten-feature CSV file"
    )

    st.write(
        "Required column order:"
    )

    st.code(
        ", ".join(
            MODEL_FEATURES
        ),
        language="text",
    )

    uploaded_batch_file = (
        st.file_uploader(
            "Select a CSV file",
            type=["csv"],
        )
    )

    if uploaded_batch_file is not None:
        try:
            uploaded_batch_data = (
                pd.read_csv(
                    uploaded_batch_file
                )
            )

            validated_batch_data = (
                validate_batch_dataframe(
                    uploaded_batch_data
                )
            )

            st.write(
                "Validated batch preview"
            )

            st.dataframe(
                validated_batch_data,
                use_container_width=True,
            )

            generate_batch_predictions = (
                st.button(
                    "Generate Batch Predictions",
                    type="primary",
                    use_container_width=True,
                )
            )

            if generate_batch_predictions:
                validated_batch_csv = (
                    validated_batch_data
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                batch_files = {
                    "file": (
                        (
                            uploaded_batch_file.name
                            or "Batch_Data_SuperKart.csv"
                        ),
                        validated_batch_csv,
                        "text/csv",
                    )
                }

                batch_response = (
                    requests.post(
                        BATCH_ENDPOINT,
                        files=batch_files,
                        timeout=(
                            REQUEST_TIMEOUT_SECONDS
                        ),
                    )
                )
                batch_result = (
                    parse_response(
                        batch_response
                    )
                )

                prediction_rows = (
                    batch_result.get(
                        "predictions",
                        [],
                    )
                )

                if (
                    len(
                        prediction_rows
                    )
                    != len(
                        validated_batch_data
                    )
                ):
                    raise RuntimeError(
                        "The backend prediction "
                        "count does not match the "
                        "input row count."
                    )

                prediction_values = [
                    float(
                        prediction_row[
                            "prediction"
                        ]
                    )
                    for prediction_row
                    in prediction_rows
                ]

                if not np.isfinite(
                    prediction_values
                ).all():
                    raise RuntimeError(
                        "The backend returned one "
                        "or more nonfinite predictions."
                    )

                batch_output = (
                    validated_batch_data
                    .copy()
                )

                batch_output[
                    "Predicted_Product_Store_"
                    "Sales_Total"
                ] = prediction_values

                st.success(
                    "Created "
                    f"{len(batch_output)} "
                    "batch predictions "
                    "successfully."
                )

                st.dataframe(
                    batch_output,
                    use_container_width=True,
                )

                batch_output_csv = (
                    batch_output
                    .to_csv(
                        index=False
                    )
                    .encode(
                        "utf-8"
                    )
                )

                st.download_button(
                    "Download Prediction Results",
                    data=batch_output_csv,
                    file_name=(
                        "superkart_batch_"
                        "predictions.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )

        except (
            requests.RequestException
        ) as error:
            st.error(
                "Could not connect to "
                "the backend: "
                f"{error}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            st.error(
                str(error)
            )
