import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

# ============================================================
# SYMCA ATTENDANCE PREDICTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="SYMCA Attendance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "symca_attendance_model.pkl"

DATA_FILES = [
    BASE_DIR / "symca_cleaned.csv",
    BASE_DIR / "symca_cleaned (1).csv"
]

DATA_PATH = next(
    (file for file in DATA_FILES if file.exists()),
    None
)

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
def load_data(path):
    data = pd.read_csv(path)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    return data


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        "❌ Model file not found: symca_attendance_model.pkl"
    )
    st.stop()

if DATA_PATH is None:
    st.error(
        "❌ Dataset file not found."
    )
    st.info(
        "Keep symca_cleaned.csv or symca_cleaned (1).csv "
        "in the same folder as app.py."
    )
    st.stop()


# ============================================================
# LOAD
# ============================================================

try:
    model = load_model()
    df = load_data(DATA_PATH)

except Exception as e:
    st.error("❌ Unable to load the model or dataset.")
    st.code(str(e))
    st.stop()


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = list(
    getattr(
        model,
        "feature_names_in_",
        []
    )
)

EXPECTED_FEATURES = [
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

    digits_part = text.split(period)[0]

    digits = re.sub(
        r"[^0-9]",
        "",
        digits_part
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

    if period == "AM" and hour == 12:
        hour = 0

    return hour * 60 + minute


def is_yes(value):

    return str(value).strip().lower() in [
        "yes",
        "true",
        "1"
    ]


def attendance_status(value):

    if value >= 75:
        return "🟢 Good"
    elif value >= 60:
        return "🟡 Moderate"
    else:
        return "🔴 Low"


def historical_features(
    section,
    subject,
    lecture_date
):

    if "Date" not in df.columns:
        return np.nan, np.nan, np.nan

    history = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Subject"].astype(str) == str(subject))
        &
        (df["Date"] < lecture_date)
    ].copy()

    if history.empty:
        return np.nan, np.nan, np.nan

    history = history.sort_values("Date")

    attendance = pd.to_numeric(
        history["Attendence Percentage"],
        errors="coerce"
    ).dropna()

    if attendance.empty:
        return np.nan, np.nan, np.nan

    previous = attendance.iloc[-1]

    rolling = attendance.tail(3).mean()

    monthly = history[
        (history["Date"].dt.year == lecture_date.year)
        &
        (history["Date"].dt.month == lecture_date.month)
    ]

    monthly_attendance = pd.to_numeric(
        monthly["Attendence Percentage"],
        errors="coerce"
    ).dropna()

    monthly_average = (
        monthly_attendance.mean()
        if not monthly_attendance.empty
        else np.nan
    )

    return previous, rolling, monthly_average


def days_since_holiday(
    section,
    lecture_date
):

    if "Holiday Before/ After" not in df.columns:
        return 0

    history = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Date"] < lecture_date)
    ].copy()

    if history.empty:
        return 0

    holidays = history[
        history["Holiday Before/ After"].apply(is_yes)
    ]

    if holidays.empty:
        return 0

    last_holiday = holidays["Date"].max()

    return max(
        0,
        (lecture_date - last_holiday).days
    )


def consecutive_lecture_count(
    section,
    lecture_date,
    start_minutes
):

    if "Start Time Minutes" not in df.columns:
        return 1

    starts = pd.to_numeric(
        df["Start Time Minutes"],
        errors="coerce"
    )

    count = len(
        df[
            (df["Section"].astype(str) == str(section))
            &
            (df["Date"] == lecture_date)
            &
            (starts < start_minutes)
        ]
    )

    return count + 1


def lecture_gap(
    section,
    lecture_date,
    start_minutes
):

    if "End Time Minutes" not in df.columns:
        return 0

    day_data = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Date"] == lecture_date)
    ].copy()

    if day_data.empty:
        return 0

    day_data["Start Time Minutes"] = pd.to_numeric(
        day_data["Start Time Minutes"],
        errors="coerce"
    )

    day_data["End Time Minutes"] = pd.to_numeric(
        day_data["End Time Minutes"],
        errors="coerce"
    )

    previous = day_data[
        day_data["Start Time Minutes"] < start_minutes
    ].sort_values("Start Time Minutes")

    if previous.empty:
        return 0

    previous_end = previous.iloc[-1]["End Time Minutes"]

    if pd.isna(previous_end):
        return 0

    return max(
        0,
        round(
            (start_minutes - previous_end) / 60,
            2
        )
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 SYMCA Attendance Prediction System"
)

st.write(
    "Predict the expected attendance percentage "
    "for a lecture using the trained machine learning model."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write(
        "**Machine Learning Problem:** Regression"
    )

    st.write(
        "**Prediction Target:** Attendance Percentage"
    )

    st.write(
        f"**Dataset Records:** {len(df):,}"
    )

    if "Date" in df.columns:

        st.write(
            f"**Dataset Start:** "
            f"{df['Date'].min().strftime('%Y-%m-%d')}"
        )

        st.write(
            f"**Dataset End:** "
            f"{df['Date'].max().strftime('%Y-%m-%d')}"
        )

    st.divider()

    st.write(
        "The model uses lecture, faculty, "
        "academic, weather and historical "
        "attendance information."
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_predict, tab_subject, tab_impact = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Predict Attendance",
        "📚 Subject Analysis",
        "🧪 Impact Analysis"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    st.header("📊 Attendance Dashboard")

    attendance = pd.to_numeric(
        df["Attendence Percentage"],
        errors="coerce"
    ).dropna()

    average_attendance = attendance.mean()
    highest_attendance = attendance.max()
    lowest_attendance = attendance.min()

    low_attendance_count = (
        attendance < 75
    ).sum()


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📈 Average Attendance",
            f"{average_attendance:.2f}%"
        )

    with col2:
        st.metric(
            "⬆️ Highest Attendance",
            f"{highest_attendance:.2f}%"
        )

    with col3:
        st.metric(
            "⬇️ Lowest Attendance",
            f"{lowest_attendance:.2f}%"
        )

    with col4:
        st.metric(
            "⚠️ Low Attendance Lectures",
            int(low_attendance_count)
        )


    st.divider()

    # --------------------------------------------------------
    # ATTENDANCE TREND
    # --------------------------------------------------------

    st.subheader(
        "📈 Attendance Trend"
    )

    trend = (
        df.groupby("Date")[
            "Attendence Percentage"
        ]
        .mean()
        .sort_index()
    )

    st.line_chart(
        trend,
        use_container_width=True
    )


    # --------------------------------------------------------
    # SECTION ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "👥 Section-wise Attendance"
    )

    section_average = (
        df.groupby("Section")[
            "Attendence Percentage"
        ]
        .mean()
        .round(2)
    )

    st.bar_chart(
        section_average,
        use_container_width=True
    )


    # --------------------------------------------------------
    # LOW ATTENDANCE
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Low Attendance Records"
    )

    low_data = df[
        pd.to_numeric(
            df["Attendence Percentage"],
            errors="coerce"
        ) < 75
    ].copy()

    if not low_data.empty:

        columns = [
            "Date",
            "Section",
            "Subject",
            "Lecture_No",
            "Attendence Percentage"
        ]

        columns = [
            col
            for col in columns
            if col in low_data.columns
        ]

        if "Date" in low_data.columns:

            low_data["Date"] = (
                low_data["Date"]
                .dt.strftime("%Y-%m-%d")
            )

        st.dataframe(
            low_data[columns]
            .sort_values(
                "Attendence Percentage"
            )
            .head(20),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PREDICTION
# ============================================================

with tab_predict:

    st.header(
        "🔮 Predict Attendance"
    )

    st.info(
        "Enter the lecture information and click "
        "**Predict Attendance**. Historical attendance "
        "features are calculated automatically."
    )


    with st.form("attendance_prediction_form"):

        # ----------------------------------------------------
        # LECTURE INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "1. Lecture Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            latest_date = (
                df["Date"].max().date()
                if "Date" in df.columns
                else pd.Timestamp.today().date()
            )

            lecture_date = st.date_input(
                "Lecture Date",
                value=latest_date
            )

        with col2:

            sections = sorted(
                df["Section"]
                .dropna()
                .astype(str)
                .unique()
            )

            section = st.selectbox(
                "Section",
                sections
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

            subjects = sorted(
                df["Subject"]
                .dropna()
                .astype(str)
                .unique()
            )

            subject = st.selectbox(
                "Subject",
                subjects
            )

        with col2:

            faculty_ids = sorted(
                df["Faculty_ID"]
                .dropna()
                .astype(str)
                .unique()
            )

            faculty_id = st.selectbox(
                "Faculty ID",
                faculty_ids
            )

        with col3:

            classrooms = sorted(
                df["Classroom"]
                .dropna()
                .astype(str)
                .unique()
            )

            classroom = st.selectbox(
                "Classroom",
                classrooms
            )


        # ----------------------------------------------------
        # SCHEDULE
        # ----------------------------------------------------

        st.subheader(
            "2. Lecture Schedule"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            start_time = st.text_input(
                "Start Time",
                "09.15 AM"
            )

        with col2:

            end_time = st.text_input(
                "End Time",
                "10.15 AM"
            )

        with col3:

            practical_theory_options = sorted(
                df["Practical/ Theory"]
                .dropna()
                .astype(str)
                .unique()
            )

            practical_theory = st.selectbox(
                "Practical / Theory",
                practical_theory_options
            )


        # ----------------------------------------------------
        # CLASS AND FACULTY
        # ----------------------------------------------------

        st.subheader(
            "3. Class & Faculty Details"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            student_values = pd.to_numeric(
                df["Total Enrolled Students"],
                errors="coerce"
            ).dropna()

            default_students = int(
                student_values.median()
            )

            total_students = st.number_input(
                "Total Enrolled Students",
                min_value=1,
                max_value=500,
                value=default_students,
                step=1
            )

        with col2:

            experience_values = pd.to_numeric(
                df["Faculty Experience"],
                errors="coerce"
            ).dropna()

            default_experience = int(
                experience_values.median()
            )

            faculty_experience = st.number_input(
                "Faculty Experience",
                min_value=0,
                max_value=50,
                value=default_experience,
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


        # ----------------------------------------------------
        # ACADEMIC / EVENT
        # ----------------------------------------------------

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


    # ========================================================
    # PREDICTION PROCESS
    # ========================================================

    if predict_button:

        lecture_date = pd.Timestamp(
            lecture_date
        )


        # ----------------------------------------------------
        # TIME CONVERSION
        # ----------------------------------------------------

        start_minutes = parse_time_to_minutes(
            start_time
        )

        end_minutes = parse_time_to_minutes(
            end_time
        )

        if pd.isna(start_minutes):

            st.error(
                "❌ Invalid start time. "
                "Example: 09.15 AM"
            )

            st.stop()

        if pd.isna(end_minutes):

            st.error(
                "❌ Invalid end time. "
                "Example: 10.15 AM"
            )

            st.stop()

        if end_minutes <= start_minutes:

            st.error(
                "❌ End time must be later "
                "than start time."
            )

            st.stop()


        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        semester_start = df["Date"].min()

        day_of_semester = (
            lecture_date - semester_start
        ).days + 1

        day_of_week = (
            lecture_date.day_name()
        )


        # ----------------------------------------------------
        # HISTORICAL FEATURES
        # ----------------------------------------------------

        previous_attendance, rolling_average, monthly_average = (
            historical_features(
                section,
                subject,
                lecture_date
            )
        )


        gap = lecture_gap(
            section,
            lecture_date,
            start_minutes
        )


        consecutive_count = consecutive_lecture_count(
            section,
            lecture_date,
            start_minutes
        )


        holiday_gap = days_since_holiday(
            section,
            lecture_date
        )


        # ----------------------------------------------------
        # LUNCH SLOT
        # ----------------------------------------------------

        if start_minutes < 780:
            lunch_slot = "Before Lunch"
        else:
            lunch_slot = "After Lunch"


        # ----------------------------------------------------
        # WEEK BEFORE EXAM
        # ----------------------------------------------------

        week_before_exam = int(
            is_yes(internal_test)
        )


        # ----------------------------------------------------
        # CREATE MODEL INPUT
        # ----------------------------------------------------

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
                previous_attendance,

            "Gap Since Previous Lecture":
                gap,

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
                holiday_gap,

            "Consecutive Lecture Count (Day)":
                int(consecutive_count),

            "Monthly Avg Attendance":
                monthly_average,

            "Rolling Avg Attendance (Prev 3)":
                rolling_average,

            "Lunch Time Slot":
                lunch_slot,

            "Week Before Exam":
                week_before_exam

        }])


        # ----------------------------------------------------
        # MODEL FEATURE ORDER
        # ----------------------------------------------------

        if MODEL_FEATURES:

            missing_features = [
                col
                for col in MODEL_FEATURES
                if col not in input_data.columns
            ]

            if missing_features:

                st.error(
                    "❌ Required model features are missing."
                )

                st.write(
                    missing_features
                )

                st.stop()

            input_data = input_data[
                MODEL_FEATURES
            ]

        else:

            input_data = input_data[
                EXPECTED_FEATURES
            ]


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

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


            expected_present = round(
                total_students
                * prediction
                / 100
            )

            expected_absent = (
                total_students
                - expected_present
            )


            status = attendance_status(
                prediction
            )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            st.success(
                "✅ Prediction generated successfully!"
            )


            st.subheader(
                "🎯 Predicted Attendance"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Expected Attendance",
                    f"{prediction:.2f}%"
                )


            with col2:

                st.metric(
                    "Attendance Status",
                    status
                )


            with col3:

                st.metric(
                    "Expected Present",
                    f"{expected_present}"
                )


            with col4:

                st.metric(
                    "Expected Absent",
                    f"{expected_absent}"
                )


            st.progress(
                int(round(prediction))
            )


            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            if prediction >= 75:

                st.success(
                    "🟢 The predicted attendance is "
                    "above the 75% attendance threshold."
                )

            elif prediction >= 60:

                st.warning(
                    "🟡 The predicted attendance is "
                    "moderate and close to the threshold."
                )

            else:

                st.error(
                    "🔴 The predicted attendance is "
                    "below the 75% attendance threshold."
                )


            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            st.subheader(
                "📋 Prediction Details"
            )


            details = pd.DataFrame({

                "Feature": [

                    "Lecture Date",
                    "Day",
                    "Subject",
                    "Section",
                    "Faculty",
                    "Classroom",
                    "Lecture Time",
                    "Practical / Theory",
                    "Total Students",
                    "Faculty Experience",
                    "Weather",
                    "Internal Test Week",
                    "Holiday Before / After",
                    "Special Event",
                    "Previous Attendance",
                    "Rolling Previous 3 Average",
                    "Monthly Average",
                    "Days Since Last Holiday"

                ],

                "Value": [

                    lecture_date.strftime(
                        "%Y-%m-%d"
                    ),

                    day_of_week,

                    subject,

                    section,

                    faculty_id,

                    classroom,

                    f"{start_time} - {end_time}",

                    practical_theory,

                    total_students,

                    faculty_experience,

                    weather,

                    internal_test,

                    holiday,

                    special_event,

                    (
                        f"{previous_attendance:.2f}%"
                        if pd.notna(previous_attendance)
                        else "Not available"
                    ),

                    (
                        f"{rolling_average:.2f}%"
                        if pd.notna(rolling_average)
                        else "Not available"
                    ),

                    (
                        f"{monthly_average:.2f}%"
                        if pd.notna(monthly_average)
                        else "Not available"
                    ),

                    holiday_gap

                ]

            })


            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # MODEL INPUT
            # ------------------------------------------------

            with st.expander(
                "🔍 View Model Input"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# SUBJECT ANALYSIS
# ============================================================

with tab_subject:

    st.header(
        "📚 Subject-wise Attendance Analysis"
    )


    subject_analysis = (
        df.groupby("Subject")[
            "Attendence Percentage"
        ]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False
        )
        .round(2)
    )


    st.bar_chart(
        subject_analysis["mean"],
        use_container_width=True
    )


    subject_table = (
        subject_analysis
        .reset_index()
    )


    subject_table.columns = [
        "Subject",
        "Average Attendance (%)",
        "Lecture Count"
    ]


    subject_table["Status"] = (
        subject_table[
            "Average Attendance (%)"
        ]
        .apply(attendance_status)
    )


    st.dataframe(
        subject_table,
        use_container_width=True,
        hide_index=True
    )


    if not subject_analysis.empty:

        lowest_subject = (
            subject_analysis["mean"]
            .idxmin()
        )

        lowest_value = (
            subject_analysis
            .loc[
                lowest_subject,
                "mean"
            ]
        )

        st.warning(
            f"⚠️ Lowest historical attendance: "
            f"**{lowest_subject}** "
            f"({lowest_value:.2f}%)"
        )


# ============================================================
# IMPACT ANALYSIS
# ============================================================

with tab_impact:

    st.header(
        "🧪 Academic & Event Impact Analysis"
    )

    st.caption(
        "These charts show historical differences "
        "in the dataset. They do not prove causation."
    )


    # --------------------------------------------------------
    # INTERNAL TEST
    # --------------------------------------------------------

    st.subheader(
        "📝 Internal Test Week"
    )

    if "Internal Test Week" in df.columns:

        test_analysis = (
            df.groupby(
                "Internal Test Week"
            )[
                "Attendence Percentage"
            ]
            .mean()
            .round(2)
        )

        st.bar_chart(
            test_analysis,
            use_container_width=True
        )


    # --------------------------------------------------------
    # HOLIDAY
    # --------------------------------------------------------

    st.subheader(
        "🏖️ Holiday Before / After"
    )

    if "Holiday Before/ After" in df.columns:

        holiday_analysis = (
            df.groupby(
                "Holiday Before/ After"
            )[
                "Attendence Percentage"
            ]
            .mean()
            .round(2)
        )

        st.bar_chart(
            holiday_analysis,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SPECIAL EVENT
    # --------------------------------------------------------

    st.subheader(
        "🎉 Special Event"
    )

    if "Special Event" in df.columns:

        event_analysis = (
            df.groupby(
                "Special Event"
            )[
                "Attendence Percentage"
            ]
            .mean()
            .round(2)
        )

        st.bar_chart(
            event_analysis,
            use_container_width=True
        )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    st.subheader(
        "🌦️ Weather-wise Attendance"
    )

    if "Weather" in df.columns:

        weather_analysis = (
            df.groupby(
                "Weather"
            )[
                "Attendence Percentage"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .round(2)
        )

        st.bar_chart(
            weather_analysis,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SYMCA Attendance Prediction System | "
    "Machine Learning Regression Model"
)
