import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

# ============================================================
# SYMCA ATTENDANCE PREDICTION - STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="SYMCA Attendance Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "symca_attendance_model.pkl"

# Your GitHub file is named:
# symca_cleaned (1).csv
DATA_PATH = BASE_DIR / "symca_cleaned (1).csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

    return df


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "Model file not found: symca_attendance_model.pkl"
    )

    st.stop()


if not DATA_PATH.exists():

    st.error(
        "Dataset file not found: symca_cleaned (1).csv"
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    model = load_model()
    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the model or dataset."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [

    "Day of Week",
    "Lecture_No",
    "Subject",
    "Faculty_ID",
    "Section",
    "Classroom",
    "Total Enrolled Students",
    "Previous Lecture Attendence",
    "Gap Since Previous Lecture",
    "Practical/ Theory",
    "Internal Test Week",
    "Holiday Before/ After",
    "Special Event",
    "Weather",
    "Faculty Experience",
    "Start Time Minutes",
    "End Time Minutes",
    "Day of Semester",
    "Days Since Last Holiday",
    "Consecutive Lecture Count (Day)",
    "Monthly Avg Attendance",
    "Rolling Avg Attendance (Prev 3)",
    "Lunch Time Slot",
    "Week Before Exam"

]

TARGET = "Attendence Percentage"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_time_to_minutes(time_text):

    if pd.isna(time_text):
        return np.nan

    text = str(time_text).strip().upper()

    if "AM" in text:
        period = "AM"

    elif "PM" in text:
        period = "PM"

    else:
        return np.nan

    text = (
        text
        .replace("AM", "")
        .replace("PM", "")
        .strip()
    )

    digits = re.sub(
        r"[^0-9]",
        "",
        text
    )

    if not digits:
        return np.nan

    try:

        if len(digits) <= 2:

            hour = int(digits)
            minute = 0

        elif len(digits) == 3:

            hour = int(digits[0])
            minute = int(digits[1:])

        else:

            hour = int(digits[:-2])
            minute = int(digits[-2:])

    except:

        return np.nan

    if hour > 12 or minute >= 60:
        return np.nan

    if period == "PM" and hour != 12:
        hour += 12

    elif period == "AM" and hour == 12:
        hour = 0

    return hour * 60 + minute


def yes_no_to_int(value):

    return int(
        str(value).strip().lower()
        in ["yes", "true", "1"]
    )


# ============================================================
# HISTORICAL FEATURES
# ============================================================

def get_history(
    section,
    subject,
    lecture_date
):

    history = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Subject"].astype(str) == str(subject))
        &
        (df["Date"] < lecture_date)
    ].copy()

    if history.empty:

        return {

            "previous_attendance": np.nan,
            "gap": 0,
            "rolling_prev3": np.nan,
            "monthly_avg": np.nan

        }

    history = history.sort_values(
        [
            "Date",
            "Start Time Minutes",
            "Lecture_No"
        ]
    )

    attendance = pd.to_numeric(
        history[TARGET],
        errors="coerce"
    )

    previous_attendance = attendance.iloc[-1]

    last_date = history["Date"].iloc[-1]

    gap = max(
        0,
        (lecture_date - last_date).days
    )

    rolling_prev3 = attendance.tail(3).mean()

    monthly_data = history[
        (history["Date"].dt.year == lecture_date.year)
        &
        (history["Date"].dt.month == lecture_date.month)
    ]

    monthly_avg = pd.to_numeric(
        monthly_data[TARGET],
        errors="coerce"
    ).mean()

    return {

        "previous_attendance":
            previous_attendance,

        "gap":
            gap,

        "rolling_prev3":
            rolling_prev3,

        "monthly_avg":
            monthly_avg

    }


# ============================================================
# DAYS SINCE LAST HOLIDAY
# ============================================================

def get_days_since_holiday(
    section,
    lecture_date
):

    history = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Date"] < lecture_date)
    ].copy()

    if history.empty:
        return np.nan

    holiday_rows = history[
        history["Holiday Before/ After"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            ["yes", "true", "1"]
        )
    ]

    if holiday_rows.empty:
        return np.nan

    last_holiday = holiday_rows[
        "Date"
    ].max()

    return max(
        0,
        (lecture_date - last_holiday).days
    )


# ============================================================
# CONSECUTIVE LECTURE COUNT
# ============================================================

def get_consecutive_lecture_count(
    section,
    lecture_date,
    start_minutes
):

    same_day = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Date"] == lecture_date)
    ].copy()

    if same_day.empty:
        return 1

    starts = pd.to_numeric(
        same_day["Start Time Minutes"],
        errors="coerce"
    )

    return int(
        (starts < start_minutes).sum() + 1
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 SYMCA Attendance Prediction System"
)

st.write(
    "Predict the expected attendance percentage "
    "for a lecture using the trained machine "
    "learning model."
)

st.info(
    "Enter the lecture details below and click "
    "`Predict Attendance`."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Model Information")

    st.write(
        "**Problem:** Regression"
    )

    st.write(
        "**Target:** Attendence Percentage"
    )

    st.write(
        "**Model:** Random Forest Regressor"
    )

    st.write(
        "**Dataset Rows:**",
        len(df)
    )

    if "Date" in df.columns:

        st.write(
            "**Dataset Start:**",
            str(df["Date"].min().date())
        )

        st.write(
            "**Dataset End:**",
            str(df["Date"].max().date())
        )


# ============================================================
# INPUT FORM
# ============================================================

with st.form(
    "attendance_prediction_form"
):

    st.subheader(
        "1. Lecture Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        prediction_date = st.date_input(
            "Lecture Date",
            value=df["Date"].max().date()
        )

    with col2:

        section_options = sorted(
            df["Section"]
            .dropna()
            .astype(str)
            .unique()
        )

        section = st.selectbox(
            "Section",
            section_options
        )

    with col3:

        lecture_no = st.number_input(
            "Lecture Number",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        subject_options = sorted(
            df["Subject"]
            .dropna()
            .astype(str)
            .unique()
        )

        subject = st.selectbox(
            "Subject",
            subject_options
        )

    with col2:

        faculty_options = sorted(
            df["Faculty_ID"]
            .dropna()
            .astype(str)
            .unique()
        )

        faculty_id = st.selectbox(
            "Faculty ID",
            faculty_options
        )

    with col3:

        classroom_options = sorted(
            df["Classroom"]
            .dropna()
            .astype(str)
            .unique()
        )

        classroom = st.selectbox(
            "Classroom",
            classroom_options
        )


    # ========================================================
    # SCHEDULE
    # ========================================================

    st.subheader(
        "2. Lecture Schedule"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        start_time = st.text_input(
            "Start Time",
            value="09.15 AM"
        )

    with col2:

        end_time = st.text_input(
            "End Time",
            value="10.15 AM"
        )

    with col3:

        practical_options = sorted(
            df["Practical/ Theory"]
            .dropna()
            .astype(str)
            .unique()
        )

        practical_theory = st.selectbox(
            "Practical / Theory",
            practical_options
        )


    # ========================================================
    # CLASS DETAILS
    # ========================================================

    st.subheader(
        "3. Class & Faculty Details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        median_students = pd.to_numeric(
            df["Total Enrolled Students"],
            errors="coerce"
        ).median()

        total_students = st.number_input(
            "Total Enrolled Students",
            min_value=1,
            max_value=500,
            value=int(median_students),
            step=1
        )

    with col2:

        median_experience = pd.to_numeric(
            df["Faculty Experience"],
            errors="coerce"
        ).median()

        faculty_experience = st.number_input(
            "Faculty Experience",
            min_value=0,
            max_value=50,
            value=int(round(median_experience)),
            step=1
        )

    with col3:

        weather_options = sorted(
            df["Weather"]
            .dropna()
            .astype(str)
            .unique()
        )

        weather = st.selectbox(
            "Weather",
            weather_options
        )


    # ========================================================
    # ACADEMIC DETAILS
    # ========================================================

    st.subheader(
        "4. Academic / Event Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        internal_test = st.selectbox(
            "Internal Test Week",
            ["No", "Yes"]
        )

    with col2:

        holiday = st.selectbox(
            "Holiday Before / After",
            ["No", "Yes"]
        )

    with col3:

        special_event = st.selectbox(
            "Special Event",
            ["No", "Yes"]
        )


    predict_button = st.form_submit_button(
        "🔮 Predict Attendance",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    lecture_date = pd.Timestamp(
        prediction_date
    )


    # ========================================================
    # CONVERT TIME
    # ========================================================

    start_minutes = parse_time_to_minutes(
        start_time
    )

    end_minutes = parse_time_to_minutes(
        end_time
    )


    if pd.isna(start_minutes):

        st.error(
            "Invalid Start Time. "
            "Example: 09.15 AM"
        )

        st.stop()


    if pd.isna(end_minutes):

        st.error(
            "Invalid End Time. "
            "Example: 10.15 AM"
        )

        st.stop()


    if end_minutes <= start_minutes:

        st.error(
            "End Time must be later "
            "than Start Time."
        )

        st.stop()


    # ========================================================
    # DAY OF SEMESTER
    # ========================================================

    semester_start = df["Date"].min()

    day_of_semester = (
        lecture_date - semester_start
    ).days + 1


    if day_of_semester < 1:

        st.error(
            "Lecture date cannot be before "
            f"{semester_start.date()}."
        )

        st.stop()


    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    day_of_week = (
        lecture_date.day_name()
    )

    lunch_time_slot = (
        "Before Lunch"
        if start_minutes < 780
        else "After Lunch"
    )


    # ========================================================
    # HISTORICAL FEATURES
    # ========================================================

    history = get_history(
        section,
        subject,
        lecture_date
    )


    days_since_holiday = (
        get_days_since_holiday(
            section,
            lecture_date
        )
    )


    consecutive_count = (
        get_consecutive_lecture_count(
            section,
            lecture_date,
            start_minutes
        )
    )


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame([{

        "Day of Week":
            day_of_week,

        "Lecture_No":
            int(lecture_no),

        "Subject":
            subject,

        "Faculty_ID":
            faculty_id,

        "Section":
            section,

        "Classroom":
            classroom,

        "Total Enrolled Students":
            int(total_students),

        "Previous Lecture Attendence":
            history[
                "previous_attendance"
            ],

        "Gap Since Previous Lecture":
            history["gap"],

        "Practical/ Theory":
            practical_theory,

        "Internal Test Week":
            internal_test,

        "Holiday Before/ After":
            holiday,

        "Special Event":
            special_event,

        "Weather":
            weather,

        "Faculty Experience":
            int(faculty_experience),

        "Start Time Minutes":
            float(start_minutes),

        "End Time Minutes":
            float(end_minutes),

        "Day of Semester":
            int(day_of_semester),

        "Days Since Last Holiday":
            days_since_holiday,

        "Consecutive Lecture Count (Day)":
            int(consecutive_count),

        "Monthly Avg Attendance":
            history["monthly_avg"],

        "Rolling Avg Attendance (Prev 3)":
            history["rolling_prev3"],

        "Lunch Time Slot":
            lunch_time_slot,

        "Week Before Exam":
            yes_no_to_int(
                internal_test
            )

    }])


    # ========================================================
    # EXACT FEATURE ORDER
    # ========================================================

    input_data = input_data[
        MODEL_FEATURES
    ]


    # ========================================================
    # PREDICT
    # ========================================================

    try:

        prediction = model.predict(
            input_data
        )[0]

        prediction = float(
            np.clip(
                prediction,
                0,
                100
            )
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            "Prediction generated successfully!"
        )

        st.subheader(
            "📊 Predicted Attendance"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Expected Attendance",
                f"{prediction:.2f}%"
            )


        with col2:

            if prediction >= 75:

                status = "Good"

            elif prediction >= 60:

                status = "Moderate"

            else:

                status = "Low"


            st.metric(
                "Attendance Status",
                status
            )


        with col3:

            st.metric(
                "Expected Absence",
                f"{100 - prediction:.2f}%"
            )


        st.progress(
            int(round(prediction))
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        if prediction >= 75:

            st.success(
                "The predicted attendance is above "
                "the 75% attendance threshold."
            )

        elif prediction >= 60:

            st.warning(
                "The predicted attendance is moderate."
            )

        else:

            st.error(
                "The predicted attendance is low."
            )


        # ====================================================
        # SHOW FEATURES
        # ====================================================

        with st.expander(
            "🔍 View calculated model features"
        ):

            feature_display = (
                input_data
                .T
                .reset_index()
            )

            feature_display.columns = [
                "Feature",
                "Value"
            ]

            st.dataframe(
                feature_display,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.warning(
            "The model and app.py must use the "
            "same feature names and preprocessing."
        )
