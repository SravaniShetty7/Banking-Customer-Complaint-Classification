import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Banking Complaint Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-header{
    text-align:center;
    padding:20px;
    border-radius:10px;
    background:#0E1117;
    color:white;
}

.result-box{
    padding:20px;
    border-radius:10px;
    border:1px solid #ddd;
    margin-top:10px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        "model_Banking_Complaint_Classifier_v1.pkl"
    )

    vectorizer = joblib.load(
        "model_tfidf_vectorizer_v1.pkl"
    )

    return model, vectorizer


model, vectorizer = load_artifacts()

# =====================================================
# CLASS LABELS
# =====================================================

CLASS_LABELS = {
    0: "Credit Card",
    1: "Credit Reporting",
    2: "Debt Collection",
    3: "Mortgages and Loans",
    4: "Retail Banking"
}

CATEGORY_ICONS = {
    "Credit Card": "💳",
    "Credit Reporting": "📊",
    "Debt Collection": "💰",
    "Mortgages and Loans": "🏠",
    "Retail Banking": "🏦"
}

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class='main-header'>
<h1>🏦 Banking Customer Complaint Classification System</h1>
<p>AI-Powered Complaint Categorization using Machine Learning & NLP</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Project Overview")

    st.info("""
This application classifies customer banking complaints into predefined categories using a Machine Learning model trained on complaint data.
""")

    st.subheader("Complaint Categories")

    st.markdown("""
- 💳 Credit Card
- 📊 Credit Reporting
- 💰 Debt Collection
- 🏠 Mortgages and Loans
- 🏦 Retail Banking
""")

    st.subheader("Technology Stack")

    st.markdown("""
- Python
- NLP
- TF-IDF Vectorization
- Scikit-Learn
- Streamlit
""")

# =====================================================
# MAIN SECTION
# =====================================================

col1, col2 = st.columns([3,1])

with col1:

    complaint = st.text_area(
        "Enter Customer Complaint",
        height=250,
        placeholder="""
Example:

My mortgage payment was increased without any prior notice from the lender and additional interest charges were applied.
"""
    )

with col2:

    st.subheader("Sample Complaints")

    st.caption("Credit Card")
    st.write(
        "My credit card was charged twice for a transaction."
    )

    st.caption("Debt Collection")
    st.write(
        "Debt collectors continue calling me even after payment."
    )

    st.caption("Credit Reporting")
    st.write(
        "A closed account is still showing as active on my credit report."
    )

# =====================================================
# PREDICTION
# =====================================================

if st.button(
    "🔍 Predict Complaint Category",
    use_container_width=True
):

    if complaint.strip() == "":

        st.warning(
            "Please enter a complaint description."
        )

    else:

        try:

            transformed_text = vectorizer.transform(
                [complaint]
            )

            prediction = model.predict(
                transformed_text
            )[0]

            category = CLASS_LABELS.get(
                prediction,
                "Unknown"
            )

            icon = CATEGORY_ICONS.get(
                category,
                "📌"
            )

            st.success(
                "Prediction completed successfully."
            )

            st.markdown("---")

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    label="Predicted Category",
                    value=f"{icon} {category}"
                )

            with col_b:

                st.metric(
                    label="Prediction Time",
                    value=datetime.now().strftime("%H:%M:%S")
                )

            # =====================================
            # PROBABILITY DISTRIBUTION
            # =====================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    transformed_text
                )[0]

                prob_df = pd.DataFrame({

                    "Category": [
                        CLASS_LABELS[i]
                        for i in range(len(probabilities))
                    ],

                    "Probability": probabilities

                })

                prob_df = prob_df.sort_values(
                    by="Probability",
                    ascending=False
                )

                st.markdown("---")

                st.subheader(
                    "Prediction Probability Distribution"
                )

                st.dataframe(
                    prob_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    prob_df.set_index("Category")
                )

            # =====================================
            # RESULT SUMMARY
            # =====================================

            st.markdown("---")

            st.subheader(
                "Complaint Analysis Summary"
            )

            summary = pd.DataFrame({

                "Attribute": [
                    "Word Count",
                    "Character Count",
                    "Predicted Category"
                ],

                "Value": [
                    len(complaint.split()),
                    len(complaint),
                    category
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {str(e)}"
            )

# =====================================================
# SAMPLE DATASET CATEGORIES
# =====================================================

st.markdown("---")

st.subheader("Supported Complaint Categories")

category_df = pd.DataFrame({

    "Category": [
        "Credit Card",
        "Credit Reporting",
        "Debt Collection",
        "Mortgages and Loans",
        "Retail Banking"
    ],

    "Examples": [

        "Fraud transactions, card charges",

        "Credit score issues, report errors",

        "Collection agency harassment",

        "Loan EMI, mortgage payment issues",

        "Savings account and transaction issues"
    ]

})

st.dataframe(
    category_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class='footer'>
Banking Customer Complaint Classification System<br>
Machine Learning | NLP | Streamlit
</div>
""", unsafe_allow_html=True)