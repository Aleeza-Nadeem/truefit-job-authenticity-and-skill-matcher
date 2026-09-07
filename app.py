"""
TrueFit — Job Posting Authenticity Checker + Skill Matcher
============================================================
Pipeline A (supervised): Random Forest fraud/authenticity classifier
Pipeline B (unsupervised): K-Means skill clustering + cosine-similarity matcher

Required files (must sit in the same folder as this app.py):
  Pipeline A:
    fraud_model_rf.pkl
    tfidf_vectorizer.pkl
    feature_cols.pkl
    industry_freq_map.pkl
    function_freq_map.pkl
    onehot_cols.pkl
    entry_75th_salary.pkl
  Pipeline B:
    skill_cluster_model.pkl
    skill_tfidf_vectorizer.pkl
    cluster_labels.pkl
    final_df_with_clusters.csv
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="TrueFit", page_icon="🧭", layout="wide")

FRAUD_THRESHOLD = 0.2  # locked in earlier from precision/recall analysis

HIGH_RISK_TITLE_TERMS = r'data entry|virtual assistant|payroll|customer service|home based|entry level|urgent'
TECH_TITLE_TERMS = r'engineer|developer|programmer|analyst|data scientist|architect'


# ============================================================
# LOAD ARTIFACTS (cached so they only load once per session)
# ============================================================
@st.cache_resource
def load_fraud_artifacts():
    model = joblib.load("fraud_model_rf.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    industry_freq_map = joblib.load("industry_freq_map.pkl")
    function_freq_map = joblib.load("function_freq_map.pkl")
    onehot_cols = joblib.load("onehot_cols.pkl")
    entry_75th = joblib.load("entry_75th_salary.pkl")
    return model, tfidf, feature_cols, industry_freq_map, function_freq_map, onehot_cols, entry_75th


@st.cache_resource
def load_skill_artifacts():
    cluster_model = joblib.load("skill_cluster_model.pkl")
    skill_tfidf = joblib.load("skill_tfidf_vectorizer.pkl")
    cluster_labels = joblib.load("cluster_labels.pkl")
    return cluster_model, skill_tfidf, cluster_labels


@st.cache_data
def load_cluster_data():
    df = pd.read_csv("final_df_with_clusters.csv")
    return df


artifacts_ok = True
try:
    (fraud_model, fraud_tfidf, feature_cols, industry_freq_map,
     function_freq_map, onehot_cols, entry_75th) = load_fraud_artifacts()
except Exception as e:
    artifacts_ok = False
    fraud_load_error = str(e)

skill_ok = True
try:
    cluster_model, skill_tfidf, cluster_labels = load_skill_artifacts()
    cluster_df = load_cluster_data()
except Exception as e:
    skill_ok = False
    skill_load_error = str(e)


# ============================================================
# HELPERS — PIPELINE A (fraud detection)
# ============================================================
def clean_text(text):
    if text is None or (isinstance(text, float) and pd.isnull(text)):
        return ''
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_fraud_feature_row(inputs):
    """
    inputs: dict with raw fields collected from the Streamlit form.
    Returns a dict of feature_name -> value, matching what feature engineering
    produced during training. Missing/unavailable pieces default sensibly.
    """
    title = inputs['title']
    company_profile = inputs['company_profile']
    has_salary = inputs['has_salary']
    salary_max = inputs['salary_max'] if has_salary else np.nan
    telecommuting = int(inputs['telecommuting'])
    has_company_logo = int(inputs['has_company_logo'])
    has_questions = int(inputs['has_questions'])
    has_department = int(inputs['has_department'])
    required_experience = inputs['required_experience']
    required_education = inputs['required_education']
    employment_type = inputs['employment_type']
    industry = inputs['industry']
    function = inputs['function']

    is_high_risk_title_combo = int(
        (telecommuting == 1) and bool(re.search(HIGH_RISK_TITLE_TERMS, title, re.IGNORECASE))
    )

    missing_all_three = int(
        (not company_profile.strip()) and (has_company_logo == 0) and (not has_salary)
    )

    high_salary_for_entry = int(
        (required_experience == 'Entry level') and has_salary and (salary_max > entry_75th)
    )

    # Can't check true duplication against the training corpus for a single
    # live submission in a meaningful way — default to 0 (not a known duplicate).
    is_duplicate_desc = 0

    lacks_screening = int(has_questions == 0)

    technical_no_education = int(
        bool(re.search(TECH_TITLE_TERMS, title, re.IGNORECASE)) and (required_education == 'Unknown')
    )

    # frequency encodings — fall back to the smallest observed frequency for unseen categories
    industry_freq = industry_freq_map.get(industry, min(industry_freq_map.values()) if industry_freq_map else 0.0)
    function_freq = function_freq_map.get(function, min(function_freq_map.values()) if function_freq_map else 0.0)
    # location_freq_map was not persisted from training — use a small neutral placeholder
    location_freq = 0.01

    feature_dict = {
        'telecommuting': telecommuting,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions,
        'has_salary': int(has_salary),
        'has_department': has_department,
        'salary_min': inputs['salary_min'] if has_salary else 0,
        'salary_max': salary_max if has_salary else 0,
        'is_high_risk_title_combo': is_high_risk_title_combo,
        'missing_all_three': missing_all_three,
        'high_salary_for_entry': high_salary_for_entry,
        'is_duplicate_desc': is_duplicate_desc,
        'lacks_screening': lacks_screening,
        'technical_no_education': technical_no_education,
        'industry_freq': industry_freq,
        'function_freq': function_freq,
        'location_freq': location_freq,
    }

    # one-hot columns
    for col in onehot_cols:
        val = 0
        if col == f'employment_type_{employment_type}':
            val = 1
        elif col == f'required_experience_{required_experience}':
            val = 1
        elif col == f'required_education_{required_education}':
            val = 1
        feature_dict[col] = val

    return feature_dict


def predict_fraud(inputs):
    feature_dict = build_fraud_feature_row(inputs)

    # structured vector, in the exact order the model was trained on
    struct_vector = np.array([[feature_dict.get(col, 0) for col in feature_cols]], dtype=float)

    combined_text = ' '.join([
        inputs['title'], inputs['company_profile'], inputs['description'],
        inputs['requirements'], inputs['benefits']
    ])
    combined_text_clean = clean_text(combined_text)
    text_vector = fraud_tfidf.transform([combined_text_clean])

    final_vector = hstack([csr_matrix(struct_vector), text_vector])

    proba = fraud_model.predict_proba(final_vector)[0, 1]
    is_fraud = int(proba >= FRAUD_THRESHOLD)
    return is_fraud, proba, feature_dict


# ============================================================
# HELPERS — PIPELINE B (skill matching)
# ============================================================
def match_user_to_cluster(user_skills_text):
    user_vec = skill_tfidf.transform([user_skills_text])
    centroids = cluster_model.cluster_centers_
    similarities = cosine_similarity(user_vec, centroids)[0]

    best_cluster = int(np.argmax(similarities))
    best_score = float(similarities[best_cluster])

    all_scores = {cluster_labels[i]: float(similarities[i]) for i in range(len(similarities))}
    all_scores_sorted = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

    return {
        'best_cluster_id': best_cluster,
        'best_match': cluster_labels[best_cluster],
        'confidence': best_score,
        'all_matches': all_scores_sorted,
    }


def get_cluster_top_skills(cluster_id, top_n=6):
    cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
    all_skills = cluster_data['skills_combined'].dropna().str.split(', ').explode()
    return all_skills.value_counts().head(top_n).index.tolist()


def skill_gap(user_skills_text, matched_cluster_id):
    user_skills = set(s.strip() for s in user_skills_text.split(',') if s.strip())
    top_skills = set(get_cluster_top_skills(matched_cluster_id))
    return list(top_skills - user_skills)


def cluster_stats(cluster_id):
    cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
    stats = {}
    if 'normalized_salary' in cluster_data.columns:
        stats['avg_salary'] = cluster_data['normalized_salary'].mean()
    if 'remote_allowed' in cluster_data.columns:
        stats['remote_pct'] = cluster_data['remote_allowed'].mean() * 100 if cluster_data['remote_allowed'].notnull().any() else None
    if 'formatted_work_type' in cluster_data.columns:
        stats['work_type_dist'] = cluster_data['formatted_work_type'].value_counts(normalize=True) * 100
    stats['count'] = len(cluster_data)
    return stats


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🧭 TrueFit")
st.sidebar.caption("Job authenticity checker + skill matcher")
page = st.sidebar.radio("Go to", ["🏠 Home", "🚩 Fraud Checker", "🎯 Skill Matcher"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "TrueFit combines two ML pipelines:\n"
    "- A supervised Random Forest model trained on labeled real/fake postings\n"
    "- An unsupervised K-Means model clustering postings by required skills\n\n"
    "Built as a portfolio project."
)


# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.title("🧭 TrueFit")
    st.subheader("Is this job posting real — and does it fit you?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚩 Fraud Checker")
        st.write(
            "Paste in a job posting's details and get a **risk flag** "
            "(likely real vs. likely fake) with a confidence score, powered by "
            "a Random Forest model trained on ~18,000 labeled real/fake postings."
        )
    with col2:
        st.markdown("### 🎯 Skill Matcher")
        st.write(
            "Enter your own skills and see which **job category cluster** you "
            "best fit into, what skills you might be missing, and typical salary "
            "/ remote-work patterns for that category — based on ~110,000 real "
            "LinkedIn postings."
        )

    st.markdown("---")
    st.info(
        "⚠️ **Honesty note:** A 'fraud' flag means this posting statistically "
        "resembles known fake postings — it is a risk signal, not proof. "
        "Similarly, the skill matcher shows current market patterns, not "
        "personalized career advice — it doesn't know your interests or aptitude."
    )


# ============================================================
# FRAUD CHECKER PAGE
# ============================================================
elif page == "🚩 Fraud Checker":
    st.title("🚩 Job Posting Fraud Checker")

    if not artifacts_ok:
        st.error(f"Could not load fraud-detection model files. Details: {fraud_load_error}")
        st.stop()

    st.write("Fill in as much detail as you have from the posting. Leave fields blank if unknown.")

    with st.form("fraud_form"):
        title = st.text_input("Job Title *", placeholder="e.g. Data Entry Clerk")
        company_profile = st.text_area("Company Profile / About the Company", height=80)
        description = st.text_area("Job Description *", height=150)
        requirements = st.text_area("Requirements", height=100)
        benefits = st.text_area("Benefits", height=80)

        c1, c2, c3 = st.columns(3)
        with c1:
            telecommuting = st.checkbox("Remote / Telecommuting allowed")
            has_company_logo = st.checkbox("Posting includes a company logo")
            has_questions = st.checkbox("Posting includes screening questions")
        with c2:
            has_department = st.checkbox("Department is specified")
            has_salary = st.checkbox("Salary range is specified")
        with c3:
            employment_type = st.selectbox(
                "Employment Type",
                ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
            )

        salary_min, salary_max = 0, 0
        if has_salary:
            sc1, sc2 = st.columns(2)
            with sc1:
                salary_min = st.number_input("Minimum salary", min_value=0, value=30000, step=1000)
            with sc2:
                salary_max = st.number_input("Maximum salary", min_value=0, value=50000, step=1000)

        c4, c5 = st.columns(2)
        with c4:
            required_experience = st.selectbox(
                "Required Experience",
                ["Unknown", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable"]
            )
            industry = st.text_input("Industry", placeholder="e.g. Information Technology and Services")
        with c5:
            required_education = st.selectbox(
                "Required Education",
                ["Unknown", "High School or equivalent", "Some College Coursework Completed",
                 "Vocational", "Associate Degree", "Bachelor's Degree", "Master's Degree",
                 "Doctorate", "Certification", "Professional", "Unspecified"]
            )
            function = st.text_input("Function", placeholder="e.g. Sales")

        submitted = st.form_submit_button("Check This Posting")

    if submitted:
        if not title.strip() or not description.strip():
            st.warning("Please provide at least a Job Title and Description.")
        else:
            inputs = dict(
                title=title, company_profile=company_profile, description=description,
                requirements=requirements, benefits=benefits,
                telecommuting=telecommuting, has_company_logo=has_company_logo,
                has_questions=has_questions, has_department=has_department,
                has_salary=has_salary, salary_min=salary_min, salary_max=salary_max,
                employment_type=employment_type, required_experience=required_experience,
                required_education=required_education, industry=industry.strip() or "Unknown",
                function=function.strip() or "Unknown",
            )

            is_fraud, proba, _ = predict_fraud(inputs)

            st.markdown("---")
            if is_fraud:
                st.error(f"⚠️ **This posting shows patterns consistent with known fake postings.**  \n"
                         f"Risk score: **{proba:.1%}** (flag threshold: {FRAUD_THRESHOLD:.0%})")
                st.write("Verify the company independently before applying or sharing any personal/financial information.")
            else:
                st.success(f"✅ **This posting looks consistent with legitimate postings.**  \n"
                           f"Risk score: **{proba:.1%}** (flag threshold: {FRAUD_THRESHOLD:.0%})")
                st.write("Still, always use your own judgment — this is a statistical signal, not a guarantee.")

            st.progress(min(float(proba), 1.0))


# ============================================================
# SKILL MATCHER PAGE
# ============================================================
elif page == "🎯 Skill Matcher":
    st.title("🎯 Skill-to-Job Category Matcher")

    if not skill_ok:
        st.error(f"Could not load skill-matching model files. Details: {skill_load_error}")
        st.stop()

    st.write(
        "Enter your skills (comma-separated). We'll match you to the closest "
        "job category cluster from ~110,000 real LinkedIn postings, and show "
        "what's typical for that category."
    )

    user_skills_text = st.text_input(
        "Your skills",
        placeholder="e.g. Sales, Business Development, Marketing"
    )

    if st.button("Find My Match") and user_skills_text.strip():
        result = match_user_to_cluster(user_skills_text)
        matched_id = result['best_cluster_id']

        st.markdown("---")
        st.subheader(f"Best fit: **{result['best_match']}**")
        st.write(f"Confidence (cosine similarity): **{result['confidence']:.2f}**")

        gap = skill_gap(user_skills_text, matched_id)
        if gap:
            st.write("**Skills common in this category that you didn't list:**")
            st.write(", ".join(gap))
        else:
            st.write("You already cover the top skills typical of this category. 🎉")

        stats = cluster_stats(matched_id)
        st.markdown("### Market snapshot for this category")
        m1, m2, m3 = st.columns(3)
        with m1:
            if stats.get('avg_salary') is not None and not pd.isnull(stats['avg_salary']):
                st.metric("Avg. normalized salary", f"${stats['avg_salary']:,.0f}")
            else:
                st.metric("Avg. normalized salary", "N/A")
        with m2:
            if stats.get('remote_pct') is not None:
                st.metric("Explicitly remote", f"{stats['remote_pct']:.0f}%")
            else:
                st.metric("Explicitly remote", "N/A")
        with m3:
            st.metric("Postings in this category", f"{stats['count']:,}")

        if 'work_type_dist' in stats:
            st.write("**Work type breakdown:**")
            st.bar_chart(stats['work_type_dist'])

        st.markdown("### How you compare across all categories")
        match_df = pd.DataFrame(
            list(result['all_matches'].items()), columns=["Category", "Similarity"]
        )
        st.bar_chart(match_df.set_index("Category"))

        st.caption(
            "This shows current market patterns, not personalized career advice — "
            "it doesn't account for your interests, aptitude, or learning curve."
        )
