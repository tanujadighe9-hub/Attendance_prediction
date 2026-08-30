import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

# ============================================================
# SYMCA ATTENDANCE PREDICTION SYSTEM
# Final Streamlit Deployment App
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
    BASE_DIR / "symca_cleaned (1).csv",
    BASE_DIR / "symca_cleaned.csv",
    BASE_DIR / "symca_cleaned (1)(1).csv"
]

DATA_PATH = next(
    (p for p in DATA_FILES if p.exists()),
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

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.sort_values(
        [
            "Date",
            "Section",
            "Start Time Minutes",
            "Lecture_No"
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# FILE CHECK
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "❌ symca_attendance_model.pkl was not found."
    )

    st.stop()


if DATA_PATH is None:

    st.error(
        "❌ Cleaned dataset was not found."
    )

    st.info(
        "Keep the CSV file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD
# ============================================================

try:

    model = load_model()
    df = load_data(DATA_PATH)

except Exception as e:

    st.error(
        "❌ Unable to load the model or dataset."
    )

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


# Exact features expected by the saved model
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
# VALIDATE MODEL
# ============================================================

if MODEL_FEATURES:

    missing_model_features = [
        col
        for col in EXPECTED_FEATURES
        if col not in MODEL_FEATURES
    ]

    if missing_model_features:

        st.error(
            "The saved model features do not match "
            "the deployment features."
        )

        st.write(
            "Missing from model:",
            missing_model_features
        )

        st.stop()


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

    except Exception:

        return np.nan

    if hour > 12 or minute >= 60:

        return np.nan

    if period == "PM" and hour != 12:

        hour += 12

    elif period == "AM" and hour == 12:

        hour = 0

    return hour * 60 + minute


def yes_value(value):

    return str(value).strip().lower() in [
        "yes",
        "true",
        "1"
    ]


def get_previous_attendance(
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
            "previous": np.nan,
            "rolling": np.nan,
            "monthly": np.nan
        }

    history = history.sort_values(
        [
            "Date",
            "Start Time Minutes",
            "Lecture_No"
        ]
    )

    attendance = pd.to_numeric(
        history["Attendence Percentage"],
        errors="coerce"
    )

    attendance = attendance.dropna()

    if attendance.empty:

        return {
            "previous": np.nan,
            "rolling": np.nan,
            "monthly": np.nan
        }

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

    monthly_avg = (
        monthly_attendance.mean()
        if not monthly_attendance.empty
        else np.nan
    )

    return {
        "previous": previous,
        "rolling": rolling,
        "monthly": monthly_avg
    }


def get_days_since_last_holiday(
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
        .apply(yes_value)
    ]

    if holiday_rows.empty:

        return np.nan

    last_holiday = holiday_rows["Date"].max()

    return (
        lecture_date - last_holiday
    ).days


def get_consecutive_count(
    section,
    lecture_date,
    start_minutes
):

    same_day = df[
        (df["Section"].astype(str) == str(section))
        &
        (df["Date"] == lecture_date)
        &
        (
            pd.to_numeric(
                df["Start Time Minutes"],
                errors="coerce"
            )
            < start_minutes
        )
    ]

    return len(same_day) + 1


def get_gap_since_previous_lecture(
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

        return 0.0

    same_day["Start Time Minutes"] = pd.to_numeric(
        same_day["Start Time Minutes"],
        errors="coerce"
    )

    same_day["End Time Minutes"] = pd.to_numeric(
        same_day["End Time Minutes"],
        errors="coerce"
    )

    previous = same_day[
        same_day["Start Time Minutes"] < start_minutes
    ].sort_values(
        "Start Time Minutes"
    )

    if previous.empty:

        return 0.0

    previous_end = previous.iloc[-1]["End Time Minutes"]

    if pd.isna(previous_end):

        return 0.0

    gap = (
        start_minutes - previous_end
    ) / 60

    return max(
        0.0,
        round(gap, 2)
    )


def attendance_status(value):

    if value >= 75:

        return "🟢 Good"

    elif value >= 60:

        return "🟡 Moderate"

    return "🔴 Low"


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎓 SYMCA Attendance Prediction System"
)

st.caption(
    "Machine Learning based Attendance Analytics "
    "and Future Lecture Prediction"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Dashboard Settings")

    low_threshold = st.slider(
        "Low Attendance Threshold (%)",
        min_value=50,
        max_value=90,
        value=75
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write(
        "Problem: **Regression**"
    )

    st.write(
        "Target: **Attendence Percentage**"
    )

    if hasattr(model, "named_steps"):

        final_estimator = model.named_steps.get(
            "model"
        )

        st.write(
            "Algorithm:",
            f"**{type(final_estimator).__name__}**"
        )

    st.divider()

    st.subheader("📁 Dataset")

    st.write(
        f"Rows: **{len(df):,}**"
    )

    st.write(
        f"Columns: **{len(df.columns)}**"
    )

    st.write(
        f"Start: **{df['Date'].min().date()}**"
    )

    st.write(
        f"End: **{df['Date'].max().date()}**"
    )


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_predict, tab_time, tab_subject, tab_impact = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Predict Attendance",
        "🕐 Time Analysis",
        "📚 Subject Analysis",
        "🧪 Impact Analysis"
    ]
)


# ============================================================
# TAB 1 - DASHBOARD
# ============================================================

with tab_dashboard:

    st.header(
        "📊 Attendance Overview"
    )

    attendance = pd.to_numeric(
        df["Attendence Percentage"],
        errors="coerce"
    )

    average_attendance = attendance.mean()
    highest_attendance = attendance.max()
    lowest_attendance = attendance.min()

    low_count = int(
        (attendance < low_threshold).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "📈 Average Attendance",
            f"{average_attendance:.2f}%"
        )

    with c2:

        st.metric(
            "⬆️ Highest Attendance",
            f"{highest_attendance:.2f}%"
        )

    with c3:

        st.metric(
            "⬇️ Lowest Attendance",
            f"{lowest_attendance:.2f}%"
        )

    with c4:

        st.metric(
            "⚠️ Low Attendance Lectures",
            low_count
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
    # ATTENDANCE BY SECTION
    # --------------------------------------------------------

    st.subheader(
        "👥 Attendance by Section"
    )

    section_avg = (
        df.groupby("Section")[
            "Attendence Percentage"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .round(2)
    )

    st.bar_chart(
        section_avg,
        use_container_width=True
    )


    # --------------------------------------------------------
    # LOW ATTENDANCE RECORDS
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Low Attendance Lectures"
    )

    low_df = df[
        df["Attendence Percentage"]
        < low_threshold
    ].copy()

    if not low_df.empty:

        display_cols = [
            "Date",
            "Section",
            "Subject",
            "Lecture_No",
            "Start_Time",
            "Attendence Percentage"
        ]

        display_cols = [
            c
            for c in display_cols
            if c in low_df.columns
        ]

        low_df["Date"] = low_df[
            "Date"
        ].dt.strftime("%Y-%m-%d")

        st.dataframe(
            low_df[
                display_cols
            ]
            .sort_values(
                "Attendence Percentage"
            )
            .head(20),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2 - PREDICTION
# ============================================================

with tab_predict:

    st.header(
        "🔮 Predict Upcoming Lecture Attendance"
    )

    st.info(
        "Enter information known before the lecture. "
        "Historical attendance features are calculated "
        "automatically from previous lectures."
    )


    with st.form(
        "prediction_form"
    ):

        st.subheader(
            "1. Lecture Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            default_date = (
                df["Date"].max().date()
                + pd.Timedelta(days=1)
            )

            lecture_date_input = st.date_input(
                "Lecture Date",
                value=default_date
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
                value=int(
                    round(median_students)
                ),
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
                value=int(
                    round(median_experience)
                ),
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


    if predict_button:

        lecture_date = pd.Timestamp(
            lecture_date_input
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        start_minutes = parse_time_to_minutes(
            start_time
        )

        end_minutes = parse_time_to_minutes(
            end_time
        )


        if pd.isna(start_minutes):

            st.error(
                "❌ Invalid Start Time. "
                "Example: 09.15 AM"
            )

            st.stop()


        if pd.isna(end_minutes):

            st.error(
                "❌ Invalid End Time. "
                "Example: 10.15 AM"
            )

            st.stop()


        if end_minutes <= start_minutes:

            st.error(
                "❌ End Time must be later "
                "than Start Time."
            )

            st.stop()


        # ----------------------------------------------------
        # DATE FEATURES
        # ----------------------------------------------------

        semester_start = df["Date"].min()

        day_of_semester = (
            lecture_date - semester_start
        ).days + 1

        if day_of_semester < 1:

            st.error(
                "Lecture date cannot be before "
                "the start of the dataset."
            )

            st.stop()


        day_of_week = (
            lecture_date.day_name()
        )


        # ----------------------------------------------------
        # HISTORICAL FEATURES
        # ----------------------------------------------------

        history = get_previous_attendance(
            section,
            subject,
            lecture_date
        )


        days_since_holiday = (
            get_days_since_last_holiday(
                section,
                lecture_date
            )
        )


        consecutive_count = (
            get_consecutive_count(
                section,
                lecture_date,
                start_minutes
            )
        )


        gap = (
            get_gap_since_previous_lecture(
                section,
                lecture_date,
                start_minutes
            )
        )


        lunch_slot = (
            "Before Lunch"
            if start_minutes < 780
            else "After Lunch"
        )


        week_before_exam = int(
            yes_value(internal_test)
        )


        # ----------------------------------------------------
        # CREATE EXACT MODEL INPUT
        # ----------------------------------------------------

        prediction_input = pd.DataFrame([{

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
                history["previous"],

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
                days_since_holiday,

            "Consecutive Lecture Count (Day)":
                int(consecutive_count),

            "Monthly Avg Attendance":
                history["monthly"],

            "Rolling Avg Attendance (Prev 3)":
                history["rolling"],

            "Lunch Time Slot":
                lunch_slot,

            "Week Before Exam":
                week_before_exam

        }])


        # ----------------------------------------------------
        # EXACT ORDER USED BY TRAINED MODEL
        # ----------------------------------------------------

        if MODEL_FEATURES:

            prediction_input = prediction_input[
                MODEL_FEATURES
            ]

        else:

            prediction_input = prediction_input[
                EXPECTED_FEATURES
            ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(
                prediction_input
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
            # RESULT
            # ------------------------------------------------

            st.success(
                "✅ Prediction generated successfully!"
            )

            st.subheader(
                "🎯 Predicted Attendance"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Expected Attendance",
                    f"{prediction:.2f}%"
                )

            with c2:

                st.metric(
                    "Status",
                    status
                )

            with c3:

                st.metric(
                    "Expected Present",
                    f"{expected_present} students"
                )

            with c4:

                st.metric(
                    "Expected Absent",
                    f"{expected_absent} students"
                )


            st.progress(
                int(round(prediction))
            )


            if prediction >= low_threshold:

                st.success(
                    f"Expected attendance is above "
                    f"the {low_threshold}% threshold."
                )

            else:

                st.warning(
                    f"⚠️ Expected attendance is below "
                    f"the {low_threshold}% threshold."
                )


            # ------------------------------------------------
            # PREDICTION DETAILS
            # ------------------------------------------------

            st.subheader(
                "📋 Prediction Details"
            )

            details = pd.DataFrame({

                "Item": [
                    "Lecture Date",
                    "Day",
                    "Subject",
                    "Section",
                    "Faculty",
                    "Lecture Time",
                    "Practical / Theory",
                    "Internal Test Week",
                    "Holiday Before / After",
                    "Weather",
                    "Previous Attendance",
                    "Rolling Previous-3 Average",
                    "Monthly Historical Average",
                    "Gap Since Previous Lecture",
                    "Consecutive Lecture Count"
                ],

                "Value": [

                    lecture_date.strftime(
                        "%Y-%m-%d"
                    ),

                    day_of_week,

                    subject,

                    section,

                    faculty_id,

                    f"{start_time} - {end_time}",

                    practical_theory,

                    internal_test,

                    holiday,

                    weather,

                    (
                        f"{history['previous']:.2f}%"
                        if pd.notna(
                            history["previous"]
                        )
                        else "Not available"
                    ),

                    (
                        f"{history['rolling']:.2f}%"
                        if pd.notna(
                            history["rolling"]
                        )
                        else "Not available"
                    ),

                    (
                        f"{history['monthly']:.2f}%"
                        if pd.notna(
                            history["monthly"]
                        )
                        else "Not available"
                    ),

                    f"{gap:.2f} hours",

                    consecutive_count

                ]

            })

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True
            )


            with st.expander(
                "🔍 View exact model input"
            ):

                st.dataframe(
                    prediction_input,
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

            st.warning(
                "The saved model expects a specific set "
                "of features. Check the model and dataset."
            )


# ============================================================
# TAB 3 - TIME ANALYSIS
# ============================================================

with tab_time:

    st.header(
        "🕐 Attendance by Time Slot"
    )

    time_df = df.copy()

    time_df["Start Time Minutes"] = pd.to_numeric(
        time_df["Start Time Minutes"],
        errors="coerce"
    )

    time_df = time_df.dropna(
        subset=["Start Time Minutes"]
    )

    time_analysis = (
        time_df.groupby(
            "Start Time Minutes"
        )[
            "Attendence Percentage"
        ]
        .mean()
        .sort_index()
        .round(2)
    )

    if not time_analysis.empty:

        time_labels = {}

        for minutes in time_analysis.index:

            hour = int(minutes // 60)
            minute = int(minutes % 60)

            if hour == 0:
                display_hour = 12
                period = "AM"

            elif hour < 12:
                display_hour = hour
                period = "AM"

            elif hour == 12:
                display_hour = 12
                period = "PM"

            else:
                display_hour = hour - 12
                period = "PM"

            time_labels[minutes] = (
                f"{display_hour:02d}:{minute:02d} {period}"
            )

        time_chart = time_analysis.rename(
            index=time_labels
        )

        st.bar_chart(
            time_chart,
            use_container_width=True
        )


        lowest_time = time_analysis.idxmin()

        highest_time = time_analysis.idxmax()

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "⚠️ Lowest Attendance Slot",
                time_labels[lowest_time],
                f"{time_analysis[lowest_time]:.2f}%"
            )

        with c2:

            st.metric(
                "🟢 Highest Attendance Slot",
                time_labels[highest_time],
                f"{time_analysis[highest_time]:.2f}%"
            )


        table = pd.DataFrame({

            "Time Slot":
                [
                    time_labels[x]
                    for x in time_analysis.index
                ],

            "Average Attendance (%)":
                time_analysis.values

        })

        table["Status"] = table[
            "Average Attendance (%)"
        ].apply(
            attendance_status
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # BEFORE / AFTER LUNCH
    # --------------------------------------------------------

    st.subheader(
        "🍱 Before Lunch vs After Lunch"
    )

    lunch_analysis = (
        df.groupby("Lunch Time Slot")[
            "Attendence Percentage"
        ]
        .mean()
        .round(2)
    )

    st.bar_chart(
        lunch_analysis,
        use_container_width=True
    )


# ============================================================
# TAB 4 - SUBJECT ANALYSIS
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

    subject_chart = subject_analysis[
        "mean"
    ]

    st.bar_chart(
        subject_chart,
        use_container_width=True
    )


    subject_table = subject_analysis.reset_index()

    subject_table.columns = [
        "Subject",
        "Average Attendance (%)",
        "Lecture Count"
    ]

    subject_table["Status"] = (
        subject_table[
            "Average Attendance (%)"
        ]
        .apply(
            attendance_status
        )
    )


    st.dataframe(
        subject_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # LOWEST SUBJECT
    # --------------------------------------------------------

    if not subject_analysis.empty:

        lowest_subject = (
            subject_analysis[
                "mean"
            ].idxmin()
        )

        lowest_value = (
            subject_analysis.loc[
                lowest_subject,
                "mean"
            ]
        )

        st.warning(
            f"⚠️ Subject with lowest historical "
            f"attendance: **{lowest_subject}** "
            f"({lowest_value:.2f}%)"
        )


# ============================================================
# TAB 5 - IMPACT ANALYSIS
# ============================================================

with tab_impact:

    st.header(
        "🧪 Academic & Event Impact Analysis"
    )

    st.caption(
        "These are historical attendance differences "
        "observed in the dataset. They should not be "
        "interpreted as guaranteed causal effects."
    )


    # --------------------------------------------------------
    # INTERNAL TEST WEEK
    # --------------------------------------------------------

    st.subheader(
        "📝 Internal Test Week"
    )

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


    if len(test_analysis) >= 2:

        try:

            normal = test_analysis["No"]
            test = test_analysis["Yes"]

            difference = test - normal

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Normal Week",
                    f"{normal:.2f}%"
                )

            with c2:

                st.metric(
                    "Internal Test Week",
                    f"{test:.2f}%"
                )

            with c3:

                st.metric(
                    "Difference",
                    f"{difference:+.2f}%"
                )

        except KeyError:

            pass


    # --------------------------------------------------------
    # HOLIDAY
    # --------------------------------------------------------

    st.subheader(
        "🏖️ Holiday Before / After"
    )

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


    if len(holiday_analysis) >= 2:

        try:

            normal_holiday = (
                holiday_analysis["No"]
            )

            holiday_value = (
                holiday_analysis["Yes"]
            )

            holiday_difference = (
                holiday_value
                - normal_holiday
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Normal",
                    f"{normal_holiday:.2f}%"
                )

            with c2:

                st.metric(
                    "Holiday Related",
                    f"{holiday_value:.2f}%"
                )

            with c3:

                st.metric(
                    "Difference",
                    f"{holiday_difference:+.2f}%"
                )

        except KeyError:

            pass


    # --------------------------------------------------------
    # SPECIAL EVENT
    # --------------------------------------------------------

    st.subheader(
        "🎉 Special Event"
    )

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
