import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from matplotlib.ticker import MaxNLocator


INDIVIDUAL_PASS_THRESHOLD = 40  
OVERALL_PASS_THRESHOLD = 140   


st.set_page_config(layout="wide", page_title="Student Performance Analysis")


@st.cache_data
def load_data(file_path):
    """Loads data and handles errors."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"ERROR: CSV file not found at '{file_path}'. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during file loading: {e}")
        return None


def analyze_data(df, individual_threshold, overall_threshold):
    """Applies the core analysis logic to the DataFrame."""
    try:
        if 'Name' not in df.columns:
            SUBJECT_COLS = df.columns.tolist()
        else:
            SUBJECT_COLS = df.columns.drop('Name').tolist()

        if not SUBJECT_COLS:
            st.error("No subject columns found after excluding 'Name'.")
            return None

    except ValueError as e:
        st.error(f"DATA ERROR: {e}")
        st.write("DataFrame columns:", df.columns.tolist())
        return None

    df['Total'] = df[SUBJECT_COLS].sum(axis=1)
    df['Passed_All_Subjects'] = (df[SUBJECT_COLS] >= individual_threshold).all(axis=1)
    df['Passed_Overall'] = (df['Total'] >= overall_threshold)

    df['Final_Result'] = df.apply(
        lambda row: 'Pass' if row['Passed_All_Subjects'] and row['Passed_Overall'] else 'Fail',
        axis=1
    )
    return df, SUBJECT_COLS



def main():
    st.title("👨‍🎓 Student Performance Dashboard")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a CSV file (e.g., student_data.csv)", type="csv")

    if uploaded_file is not None:
        df_raw = load_data(uploaded_file)
    else:
        st.stop() 

    if df_raw is None:
        st.stop() 

    df = df_raw.copy()

    with st.sidebar:
        st.header("Analysis Configuration")
        ind_thresh = st.slider(
            "Individual Subject Pass Threshold",
            min_value=1, max_value=100, value=INDIVIDUAL_PASS_THRESHOLD, step=1
        )
        ovr_thresh = st.slider(
            "Overall Total Pass Threshold",
            min_value=50, max_value=500, value=OVERALL_PASS_THRESHOLD, step=10
        )
        st.info(f"Pass: Individual $\\ge {ind_thresh}$ AND Total $\\ge {ovr_thresh}$")

    analysis_result = analyze_data(df, ind_thresh, ovr_thresh)

    if analysis_result is None:
        st.stop()

    df_analyzed, SUBJECT_COLS = analysis_result

    st.header("📊 Final Results Overview")
    st.info(f"Analysis based on: Individual Subject $\\ge {ind_thresh}$ and Total $\\ge {ovr_thresh}$")

    st.dataframe(df_analyzed, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Total Marks")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_analyzed['Total'], bins=5, kde=True, color='#4c72b0', ax=ax)
        ax.set_title('Distribution of Total Marks', fontsize=14)
        ax.set_xlabel('Total Marks', fontsize=10)
        ax.set_ylabel('Number of Students', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        st.pyplot(fig)

    with col2:
        st.subheader("Final Pass vs Fail Count")
        result_counts = df_analyzed['Final_Result'].value_counts()
        colors = {'Pass': '#66c2a5', 'Fail': '#fc8d62'}
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(result_counts.index, result_counts.values,
                      color=[colors.get(label, 'gray') for label in result_counts.index])
        ax.set_title('Final Pass vs Fail Count', fontsize=14)
        ax.set_xlabel('Result', fontsize=10)
        ax.set_ylabel('Number of Students', fontsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                      f'{int(height)}',
                      ha='center', va='bottom', fontsize=10)

        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Raw Data Summary")
    st.write(f"Total students analyzed: **{len(df_analyzed)}**")
    st.write(f"Subject columns detected: **{', '.join(SUBJECT_COLS)}**")


    st.markdown("---")
    st.header("🧮 Predict Result for a New Student")

    st.write("Enter marks for each subject to predict whether the student will Pass or Fail.")

    new_student = {}
    for subject in SUBJECT_COLS:
        new_student[subject] = st.number_input(
            f"Enter marks for {subject}",
            min_value=0,
            max_value=1000,
            value=0
        )

    if st.button("Predict Result"):
        total_marks = sum(new_student.values())
        passed_all = all(mark >= ind_thresh for mark in new_student.values())
        passed_total = total_marks >= ovr_thresh

        final_result = "Pass" if (passed_all and passed_total) else "Fail"

        st.subheader("Prediction Result")
        st.write(f"**Total Marks:** {total_marks}")
        st.write(f"**Passed All Subjects:** {passed_all}")
        st.write(f"**Passed Overall Total:** {passed_total}")
        st.success(f"🎉 Final Prediction: **{final_result}**" if final_result == "Pass" else f"❌ Final Prediction: **{final_result}**")


if __name__ == '__main__':
    main()
