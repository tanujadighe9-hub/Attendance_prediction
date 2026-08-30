"""
SYMCA Attendance Prediction Dashboard
--------------------------------------
A simple Streamlit app that:
  1. Loads the trained model (symca_attendance_model.pkl)
  2. Lets a faculty member enter details of an UPCOMING lecture and get a
     predicted attendance percentage
  3. Shows a dashboard of historical attendance patterns (by subject, day,
     test week, etc.) so department heads can spot low-attendance trends

To run this app:
    streamlit run app.py

Make sure these two files are in the SAME folder as app.py:
    - symca_attendance_model.pkl
    - symca_cleaned.csv
"""

import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SYMCA Attendance Predictor", page_icon="\U0001F4CA", layout="wide")

st.title("\U0001F4CA SYMCA Attendance Prediction Dashboard")
st.caption("DSML Capstone \u2014 predicts attendance % for upcoming lectures using historical patterns")


# ---------------------------------------------------------------------------
# LOAD MODEL AND DATA (cached so it only loads once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("symca_attendance_model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("symca_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


model = load_model()
df = load_data()

# Build a lookup table: for each Subject, what Classroom / Practical-or-Theory
# type / typical Faculty does it normally use? This lets the app auto-fill
# those fields instead of asking the user to type them every time.
def most_common(series):
    mode = series.dropna().mode()
    return mode.iloc[0] if len(mode) > 0 else None

subject_lookup = (
    df.groupby("Subject")
    .agg(
        {
            "Classroom": most_common,
            "Practical/ Theory": most_common,
            "Faculty_ID": most_common,
            "Total Enrolled Students": most_common,
            "Faculty Experience": "mean",
        }
    )
    .to_dict(orient="index")
)

subject_list = sorted(subject_lookup.keys())
overall_avg_experience = df["Faculty Experience"].mean()


# ---------------------------------------------------------------------------
# TABS: Predict | Dashboard
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["\U0001F52E Predict Attendance", "\U0001F4C8 Attendance Dashboard"])


# ===========================================================================
# TAB 1: PREDICT ATTENDANCE FOR AN UPCOMING LECTURE
# ===========================================================================
with tab1:
    st.subheader("Enter details of the upcoming lecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        subject = st.selectbox("Subject", subject_list)
        section = st.selectbox("Section", sorted(df["Section"].unique()))
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        )
        lecture_number = st.number_input("Lecture Number (slot in the day)", min_value=1, max_value=7, value=1)

    with col2:
        start_time = st.time_input("Start Time")
        end_time = st.time_input("End Time")
        weather = st.selectbox("Weather", sorted(df["Weather"].unique()))
        previous_attendance = st.slider("Previous Lecture Attendance (%)", 0.0, 100.0, 75.0)

    with col3:
        internal_test_week = st.selectbox("Internal Test Week?", ["No", "Yes"])
        holiday_before_after = st.selectbox("Holiday Before/After?", ["No", "Yes"])
        special_event = st.selectbox("Special Event?", ["No", "Yes"])
        gap_since_previous = st.number_input("Gap Since Previous Lecture (hrs)", min_value=0.0, value=0.0, step=0.5)

    st.markdown("**Additional context** (auto-calculated fields \u2014 adjust if needed)")
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        days_since_holiday = st.number_input("Days Since Last Holiday", min_value=0, value=10)
    with col5:
        rolling_avg = st.slider("Rolling Avg Attendance (Prev 3 Lectures)", 0.0, 100.0, 75.0)
    with col6:
        monthly_avg = st.slider("Monthly Avg Attendance", 0.0, 100.0, 75.0)
    with col7:
        day_of_semester = st.number_input("Day of Semester", min_value=1, max_value=200, value=30)

    consecutive_lectures = st.number_input("Consecutive Lectures That Day", min_value=1, max_value=10, value=3)
    week_before_exam = st.selectbox("Is this within a week before an exam?", ["No", "Yes"])

    st.divider()

    if st.button("\U0001F52E Predict Attendance", type="primary"):
        info = subject_lookup[subject]

        start_minutes = start_time.hour * 60 + start_time.minute
        end_minutes = end_time.hour * 60 + end_time.minute
        lunch_slot = "Before Lunch" if start_time.hour < 13 else "After Lunch"

        faculty_experience = info["Faculty Experience"]
        if pd.isna(faculty_experience):
            faculty_experience = overall_avg_experience

        faculty_id = info["Faculty_ID"]
        if faculty_id is None:
            faculty_id = "GUEST"

        def build_row(test_week_value, holiday_value):
            """Build one input row for the model. Reused to simulate
            'what if this were test week / holiday' scenarios below."""
            return pd.DataFrame([{
                "Lecture_No": lecture_number,
                "Total Enrolled Students": info["Total Enrolled Students"],
                "Previous Lecture Attendence": previous_attendance,
                "Gap Since Previous Lecture": gap_since_previous,
                "Faculty Experience": faculty_experience,
                "Start Time Minutes": start_minutes,
                "End Time Minutes": end_minutes,
                "Day of Semester": day_of_semester,
                "Days Since Last Holiday": days_since_holiday,
                "Consecutive Lecture Count (Day)": consecutive_lectures,
                "Monthly Avg Attendance": monthly_avg,
                "Rolling Avg Attendance (Prev 3)": rolling_avg,
                "Day of Week": day_of_week,
                "Subject": subject,
                "Faculty_ID": faculty_id,
                "Section": section,
                "Classroom": info["Classroom"],
                "Practical/ Theory": info["Practical/ Theory"],
                "Internal Test Week": test_week_value,
                "Holiday Before/ After": holiday_value,
                "Special Event": special_event,
                "Weather": weather,
                "Lunch Time Slot": lunch_slot,
                "Week Before Exam": week_before_exam,
            }])

        input_row = build_row(internal_test_week, holiday_before_after)
        prediction = model.predict(input_row)[0]
        prediction = max(0, min(100, prediction))

        if prediction < 50:
            risk_label, risk_color = "Low Attendance Risk", "red"
        elif prediction < 75:
            risk_label, risk_color = "Medium Attendance", "orange"
        else:
            risk_label, risk_color = "Good Attendance Expected", "green"

        st.success("Prediction complete!")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Attendance", f"{prediction:.1f}%")
        with res_col2:
            st.markdown(f"### :{risk_color}[{risk_label}]")

        # -------------------------------------------------------------
        # "What-if" impact estimator (PDF Section 6.1 requirement:
        # estimate the impact of upcoming tests / holidays)
        # -------------------------------------------------------------
        st.divider()
        st.markdown("#### \U0001F52C Estimated impact of Test Week / Holiday on this lecture")

        normal_pred = model.predict(build_row("No", "No"))[0]
        test_week_pred = model.predict(build_row("Yes", "No"))[0]
        holiday_pred = model.predict(build_row("No", "Yes"))[0]

        impact_col1, impact_col2, impact_col3 = st.columns(3)
        impact_col1.metric("If Normal Week", f"{max(0, min(100, normal_pred)):.1f}%")
        impact_col2.metric(
            "If Test Week",
            f"{max(0, min(100, test_week_pred)):.1f}%",
            delta=f"{test_week_pred - normal_pred:+.1f} pts",
        )
        impact_col3.metric(
            "If Holiday Before/After",
            f"{max(0, min(100, holiday_pred)):.1f}%",
            delta=f"{holiday_pred - normal_pred:+.1f} pts",
        )
        st.caption("Delta shown compares against a normal week with no holiday nearby, holding all other inputs fixed.")


# ===========================================================================
# TAB 2: DASHBOARD OF HISTORICAL PATTERNS
# ===========================================================================
with tab2:
    st.subheader("Historical Attendance Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Lectures Recorded", len(df))
    k2.metric("Average Attendance", f"{df['Attendence Percentage'].mean():.1f}%")
    k3.metric("Lowest Attendance", f"{df['Attendence Percentage'].min():.1f}%")
    k4.metric("Highest Attendance", f"{df['Attendence Percentage'].max():.1f}%")

    st.markdown("### Average Attendance by Subject")
    subject_avg = df.groupby("Subject")["Attendence Percentage"].mean().sort_values()
    st.bar_chart(subject_avg)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Average Attendance by Day of Week")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        day_avg = df.groupby("Day of Week")["Attendence Percentage"].mean().reindex(day_order)
        st.bar_chart(day_avg)

    with col_b:
        st.markdown("### Test Week vs Normal Week")
        test_avg = df.groupby("Internal Test Week")["Attendence Percentage"].mean()
        st.bar_chart(test_avg)

    st.markdown("### Attendance Trend Over the Semester")
    daily_avg = df.groupby("Date")["Attendence Percentage"].mean()
    st.line_chart(daily_avg)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("### Average Attendance by Time Slot")
        lunch_avg = df.groupby("Lunch Time Slot")["Attendence Percentage"].mean()
        st.bar_chart(lunch_avg)
    with col_d:
        st.markdown("### Holiday Proximity Impact")
        holiday_avg = df.groupby("Holiday Before/ After")["Attendence Percentage"].mean()
        st.bar_chart(holiday_avg)

    st.markdown("### \u26A0\uFE0F Subjects With Consistently Low Attendance (below 70%)")
    low_subjects = subject_avg[subject_avg < 70]
    if len(low_subjects) > 0:
        st.dataframe(low_subjects.reset_index().rename(columns={"Attendence Percentage": "Avg Attendance %"}))
    else:
        st.info("No subject currently averages below 70% attendance.")

    st.markdown("### Raw Data (Cleaned Dataset)")
    st.dataframe(df.head(50))
