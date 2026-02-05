# import pandas as pd
# from pathlib import Path
# from joblib import load
#
# # Constants & Paths
# BASE_PATH = Path("artifacts")
# FEATURE_COLUMNS_PATH = BASE_PATH / "feature_columns.joblib"
# MODEL_YOUNG_PATH = BASE_PATH / "model_young.joblib"
# MODEL_REST_PATH = BASE_PATH / "model_rest.joblib"
# SCALER_YOUNG_PATH = BASE_PATH / "scaler_young.joblib"
# SCALER_REST_PATH = BASE_PATH / "scaler_rest.joblib"
#
# # Load artifacts
# feature_columns = load(FEATURE_COLUMNS_PATH)
# model_young = load(MODEL_YOUNG_PATH)
# model_rest = load(MODEL_REST_PATH)
# scaler_young = load(SCALER_YOUNG_PATH)
# scaler_rest = load(SCALER_REST_PATH)
#
#
# def calculate_normalized_risk(medical_history: str) -> float:
#     risk_scores = {
#         "diabetes": 6, "heart disease": 8, "high blood pressure": 6,
#         "thyroid": 5, "no disease": 0, "none": 0
#     }
#
#     # MATCHING INSTRUCTOR LOGIC: Lowercase then split by literal " & "
#     # Note: If your input is "Diabetes&Heart", this split will fail to find them.
#     diseases = medical_history.lower().split(" & ")
#
#     total_score = sum(risk_scores.get(d, 0) for d in diseases)
#
#     max_score = 14
#     min_score = 0
#
#     # Exact mathematical formula used by instructor
#     return (total_score - min_score) / (max_score - min_score)
#
#
# def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
#     scaler_object = scaler_young if age <= 25 else scaler_rest
#     cols_to_scale = scaler_object["cols_to_scale"]
#     scaler = scaler_object["scaler"]
#
#     temp = df.copy()
#
#     # MATCHING INSTRUCTOR LOGIC: Use None instead of 0
#     temp["income_level"] = None
#
#     temp[cols_to_scale] = scaler.transform(temp[cols_to_scale])
#     return temp.drop(columns=["income_level"])
#
#
# def preprocess_input(input_dict: dict) -> pd.DataFrame:
#     # Use the loaded feature_columns to ensure order is identical to training
#     df = pd.DataFrame(0, columns=feature_columns, index=[0])
#
#     # Mapping logic (Your clean version, but ensure keys match instructor's input)
#     one_hot_map = {
#         "Gender": {"Male": "gender_Male"},
#         "Region": {"Northwest": "region_Northwest", "Southeast": "region_Southeast", "Southwest": "region_Southwest"},
#         "Marital Status": {"Unmarried": "marital_status_Unmarried"},
#         "BMI Category": {"Obesity": "bmi_category_Obesity", "Overweight": "bmi_category_Overweight",
#                          "Underweight": "bmi_category_Underweight"},
#         "Smoking Status": {"Occasional": "smoking_status_Occasional", "Regular": "smoking_status_Regular"},
#         "Employment Status": {"Salaried": "employment_status_Salaried",
#                               "Self-Employed": "employment_status_Self-Employed"}
#     }
#
#     # Encode Categorical
#     for key, value in input_dict.items():
#         if key in one_hot_map and value in one_hot_map[key]:
#             df[one_hot_map[key][value]] = 1
#
#     # Assign Numeric Values
#     plan_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
#     df["insurance_plan"] = plan_map.get(input_dict.get("Insurance Plan"), 1)
#     df["age"] = input_dict.get("Age", 0)
#     df["number_of_dependants"] = input_dict.get("Number of Dependants", 0)
#     df["income_lakhs"] = input_dict.get("Income in Lakhs", 0)
#     df["genetical_risk"] = input_dict.get("Genetical Risk", 0)
#
#     # Risk Score
#     df["normalized_risk_score"] = calculate_normalized_risk(input_dict.get("Medical History", ""))
#
#     # Scaling
#     df = handle_scaling(input_dict["Age"], df)
#     return df
#
# def predict(input_dict: dict) -> int:
#     input_df = preprocess_input(input_dict)
#     model = model_young if input_dict["Age"] <= 25 else model_rest
#     return int(model.predict(input_df)[0])
#
#
#

# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path
import pandas as pd
import joblib

BASE_PATH = Path("artifacts")

MODEL_YOUNG_PATH = BASE_PATH / "model_young.joblib"
MODEL_REST_PATH  = BASE_PATH / "model_rest.joblib"

SCALER_YOUNG_PATH = BASE_PATH / "scaler_young.joblib"
SCALER_REST_PATH  = BASE_PATH / "scaler_rest.joblib"

INSURANCE_PLAN_ENCODING = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3
}

RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0
}

MAX_RISK_SCORE = 14


ONE_HOT_MAP = {
    "Gender": {"Male": "gender_Male"},

    "Region": {
        "Northwest": "region_Northwest",
        "Southeast": "region_Southeast",
        "Southwest": "region_Southwest",
    },

    "Marital Status": {
        "Unmarried": "marital_status_Unmarried"
    },

    "BMI Category": {
        "Obesity": "bmi_category_Obesity",
        "Overweight": "bmi_category_Overweight",
        "Underweight": "bmi_category_Underweight",
    },

    "Smoking Status": {
        "Occasional": "smoking_status_Occasional",
        "Regular": "smoking_status_Regular",
    },

    "Employment Status": {
        "Salaried": "employment_status_Salaried",
        "Self-Employed": "employment_status_Self-Employed",
    }
}

REQUIRED_FIELDS = [
    "Age",
    "Medical History",
    "Income in Lakhs"
]

# ============================================================
# LOAD ARTIFACTS
# ============================================================

model_young = joblib.load(MODEL_YOUNG_PATH)
model_rest  = joblib.load(MODEL_REST_PATH)

scaler_young = joblib.load(SCALER_YOUNG_PATH)
scaler_rest  = joblib.load(SCALER_REST_PATH)


# ============================================================
# VALIDATION
# ============================================================

def validate_input(input_dict: dict) -> None:
    for field in REQUIRED_FIELDS:
        if field not in input_dict:
            raise ValueError(f"Missing required field: {field}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_normalized_risk(medical_history: str) -> float:
    diseases = [
        d.strip().lower()
        for d in medical_history.replace("&", ",").split(",")
    ]

    total_score = sum(RISK_SCORES.get(d, 0) for d in diseases)

    return total_score / MAX_RISK_SCORE


# ============================================================
# DATAFRAME CREATION
# ============================================================

def create_empty_dataframe() -> pd.DataFrame:
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
        'genetical_risk', 'normalized_risk_score',

        'gender_Male',

        'region_Northwest', 'region_Southeast', 'region_Southwest',

        'marital_status_Unmarried',

        'bmi_category_Obesity',
        'bmi_category_Overweight',
        'bmi_category_Underweight',

        'smoking_status_Occasional',
        'smoking_status_Regular',

        'employment_status_Salaried',
        'employment_status_Self-Employed'
    ]

    return pd.DataFrame(0, columns=expected_columns, index=[0])


# ============================================================
# ENCODING LOGIC
# ============================================================

def encode_categoricals(df: pd.DataFrame, input_dict: dict) -> None:
    for key, value in input_dict.items():
        if key in ONE_HOT_MAP and value in ONE_HOT_MAP[key]:
            df[ONE_HOT_MAP[key][value]] = 1


def assign_numeric_values(df: pd.DataFrame, input_dict: dict) -> None:
    df["insurance_plan"] = INSURANCE_PLAN_ENCODING.get(
        input_dict.get("Insurance Plan"),
        1
    )

    df["age"] = input_dict.get("Age", 0)
    df["number_of_dependants"] = input_dict.get("Number of Dependants", 0)
    df["income_lakhs"] = input_dict.get("Income in Lakhs", 0)
    df["genetical_risk"] = input_dict.get("Genetical Risk", 0)


def add_risk_feature(df: pd.DataFrame, input_dict: dict) -> None:
    df["normalized_risk_score"] = calculate_normalized_risk(
        input_dict.get("Medical History", "")
    )


# ============================================================
# SCALING
# ============================================================

def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    temp = df.copy()

    # dummy column expected by scaler (kept exactly like instructor logic)
    temp["income_level"] = 0

    temp[cols_to_scale] = scaler.transform(temp[cols_to_scale])

    return temp.drop(columns=["income_level"])


# ============================================================
# PREPROCESS PIPELINE (MANUAL – Instructor Style)
# ============================================================

def preprocess_input(input_dict: dict) -> pd.DataFrame:
    validate_input(input_dict)

    df = create_empty_dataframe()

    encode_categoricals(df, input_dict)
    assign_numeric_values(df, input_dict)
    add_risk_feature(df, input_dict)

    df = handle_scaling(input_dict["Age"], df)

    return df


# ============================================================
# PREDICTION
# ============================================================

def predict(input_dict: dict) -> int:
    input_df = preprocess_input(input_dict)

    if input_dict["Age"] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
