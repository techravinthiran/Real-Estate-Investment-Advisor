import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a7fa3;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .good-investment {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 4px solid #2e7d32;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1b5e20;
    }
    .bad-investment {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 4px solid #e65100;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #bf360c;
    }
    .future-price-card {
        background: linear-gradient(135deg, #ede7f6, #d1c4e9);
        border-left: 4px solid #6a1b9a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a148c;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3c5e;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #e3eaf3;
        padding-bottom: 0.3rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        width: 100%;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load Models
@st.cache_resource
def load_models():
    try:
        clf      = joblib.load('model/best_classifier.pkl')
        reg      = joblib.load('model/best_regressor.pkl')
        scaler   = joblib.load('model/scaler.pkl')
        encoders = joblib.load('model/label_encoders.pkl')
        meta     = joblib.load('model/meta.pkl')
        features = meta['features']
        return clf, reg, scaler, encoders, meta, features, True
    except Exception as e:
        return None, None, None, None, None, None, False

clf, reg, scaler, encoders, meta, FEATURES, models_loaded = load_models()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('dataset/processed_housing.csv')
    except:
        return None

df = load_data()

# Header
st.markdown('<div class="main-header">🏠 Real Estate Investment Advisor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict Property Profitability & Future Value using Machine Learning</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("""
    Models not found! Please run apps.ipynb first to train and save the models.

    Expected files in ./model/:
    - best_classifier.pkl
    - best_regressor.pkl
    - scaler.pkl
    - label_encoders.pkl
    - meta.pkl
    """)
    st.stop()

# Sidebar Navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Predict Investment", "Data Insights"])

# Predict Investment
if page == "Predict Investment":

    st.markdown('<div class="section-header">Enter Property Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Location**")
        state = st.selectbox("State", meta['states'])
        city  = st.selectbox("City", meta['cities'])

        st.markdown("**Property**")
        prop_type   = st.selectbox("Property Type", meta['property_types'])
        bhk         = st.slider("BHK", 1, 6, 2)
        size_sqft   = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1500, step=50)
        price_lakhs = st.number_input("Price (Lakhs)", min_value=5.0, max_value=5000.0, value=80.0, step=5.0)

    with col2:
        st.markdown("**Building Details**")
        year_built   = st.slider("Year Built", 1970, 2024, 2010)
        floor_no     = st.slider("Floor No", 0, 50, 3)
        total_floors = st.slider("Total Floors", 1, 60, 10)
        furnished    = st.selectbox("Furnished Status", meta['furnished_statuses'])
        facing       = st.selectbox("Facing", meta['facings'])

        st.markdown("**Amenities**")
        amenity_count = st.slider("Number of Amenities", 0, 6, 3)
        has_parking   = st.selectbox("Parking Space", ["Yes", "No"])
        has_security  = st.selectbox("Security", ["Yes", "No"])

    with col3:
        st.markdown("**Nearby Facilities**")
        nearby_schools   = st.slider("Nearby Schools", 0, 10, 5)
        nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 3)
        transport        = st.selectbox("Public Transport", meta['transport_options'])

        st.markdown("**Ownership**")
        owner_type   = st.selectbox("Owner Type", meta['owner_types'])
        availability = st.selectbox("Availability Status", meta['availability_statuses'])

    st.markdown("---")
    predict_btn = st.button("Predict Now")

    if predict_btn:
        # Feature Engineering
        age_of_property   = 2025 - year_built
        price_per_sqft    = price_lakhs / max(size_sqft, 1)
        transport_map_num = {'Low': 1, 'Medium': 2, 'High': 3}
        transport_score   = transport_map_num.get(transport, 2)
        infrastructure_sc = nearby_schools * 0.3 + nearby_hospitals * 0.3 + transport_score * 0.4
        floor_ratio       = floor_no / max(total_floors, 1)
        parking_bin       = 1 if has_parking == "Yes" else 0
        security_bin      = 1 if has_security == "Yes" else 0

        def safe_encode(enc_dict, col, val):
            le = enc_dict.get(col)
            if le is None:
                return 0
            classes = list(le.classes_)
            return classes.index(str(val)) if str(val) in classes else 0

        row = {
            'BHK':                     bhk,
            'Size_in_SqFt':            size_sqft,
            'Price_per_SqFt':          price_per_sqft,
            'Age_of_Property':         age_of_property,
            'Floor_No':                floor_no,
            'Total_Floors':            total_floors,
            'Nearby_Schools':          nearby_schools,
            'Nearby_Hospitals':        nearby_hospitals,
            'Amenity_Count':           amenity_count,
            'Infrastructure_Score':    infrastructure_sc,
            'Has_Parking':             parking_bin,
            'Has_Security':            security_bin,
            'Floor_Ratio':             floor_ratio,
            'Transport_Score':         transport_score,
            'Furnished_Status_Enc':    safe_encode(encoders, 'Furnished_Status', furnished),
            'Facing_Enc':              safe_encode(encoders, 'Facing', facing),
            'Owner_Type_Enc':          safe_encode(encoders, 'Owner_Type', owner_type),
            'Availability_Status_Enc': safe_encode(encoders, 'Availability_Status', availability),
            'Property_Type_Enc':       safe_encode(encoders, 'Property_Type', prop_type),
            'State_Enc':               safe_encode(encoders, 'State', state),
            'City_Enc':                safe_encode(encoders, 'City', city),
        }

        input_df = pd.DataFrame([{f: row.get(f, 0) for f in FEATURES}])

        clf_name = meta.get('best_clf_name', '')
        reg_name = meta.get('best_reg_name', '')

        if clf_name in ['Logistic Regression', 'KNN']:
            clf_input_arr = scaler.transform(input_df)
        else:
            clf_input_arr = input_df.values

        if reg_name in ['Linear Regression', 'Ridge Regression']:
            reg_input_arr = scaler.transform(input_df)
        else:
            reg_input_arr = input_df.values

        prediction_clf = clf.predict(clf_input_arr)[0]
        confidence_clf = clf.predict_proba(clf_input_arr)[0]
        prediction_reg = reg.predict(reg_input_arr)[0]

        st.markdown("---")
        st.markdown("## Prediction Results")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            confidence_pct = confidence_clf[prediction_clf] * 100
            if prediction_clf == 1:
                st.markdown(f"""
                <div class="good-investment">
                    GOOD INVESTMENT<br>
                    <small>Confidence: {confidence_pct:.1f}% | Model: {clf_name}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bad-investment">
                    NOT A GOOD INVESTMENT<br>
                    <small>Confidence: {confidence_pct:.1f}% | Model: {clf_name}</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Confidence**")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(['Not Good', 'Good Investment'], [confidence_clf[0], confidence_clf[1]],
                    color=['#ef5350', '#66bb6a'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Classification Confidence')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with res_col2:
            growth = prediction_reg - price_lakhs
            growth_pct = (growth / price_lakhs) * 100
            st.markdown(f"""
            <div class="future-price-card">
                Estimated Price After 5 Years<br>
                <span style="font-size:1.5rem;">Rs. {prediction_reg:.2f} Lakhs</span><br>
                <small>Current: Rs.{price_lakhs:.2f}L | Growth: +Rs.{growth:.2f}L ({growth_pct:.1f}%) | Model: {reg_name}</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            years_list = list(range(0, 6))
            prices     = [price_lakhs * (1.08 ** y) for y in years_list]
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.plot(years_list, prices, marker='o', color='#7b1fa2', linewidth=2)
            ax2.fill_between(years_list, price_lakhs, prices, alpha=0.15, color='#9c27b0')
            ax2.set_xlabel('Years from Now')
            ax2.set_ylabel('Price (Lakhs)')
            ax2.set_title('5-Year Price Projection (8% p.a.)')
            for y, p in zip(years_list, prices):
                ax2.annotate(f'Rs.{p:.0f}', (y, p), textcoords='offset points',
                             xytext=(0, 8), ha='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.markdown("### Property Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price",   f"Rs.{price_lakhs:.1f}L")
        m2.metric("Price/SqFt",      f"Rs.{price_per_sqft:.3f}")
        m3.metric("Future Price",    f"Rs.{prediction_reg:.1f}L", delta=f"+Rs.{growth:.1f}L")
        m4.metric("Good Investment", "Yes" if prediction_clf == 1 else "No",
                  delta=f"{confidence_clf[prediction_clf]*100:.1f}% confidence")

# Data Insights
elif page == "Data Insights":
    st.markdown("## Data Insights & EDA")

    if df is None:
        st.warning("Processed dataset not found. Please run apps.ipynb first.")
        st.stop()

    st.markdown("### Dataset Overview")
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Properties", f"{len(df):,}")
    ov2.metric("States",           df['State'].nunique())
    ov3.metric("Cities",           df['City'].nunique())
    if 'Good_Investment' in df.columns:
        ov4.metric("Good Investments", f"{df['Good_Investment'].mean():.1%}")

    with st.expander("View Sample Data"):
        st.dataframe(df.head(20))

    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Location Analysis", "Correlations"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df['Price_in_Lakhs'], bins=50, color='steelblue', edgecolor='white')
            ax.set_title('Price Distribution (Lakhs)')
            ax.set_xlabel('Price in Lakhs'); ax.set_ylabel('Count')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='Furnished_Status', y='Price_in_Lakhs', palette='pastel', ax=ax)
            ax.set_title('Price by Furnished Status')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values(ascending=False).plot(
                kind='bar', ax=ax, color=sns.color_palette('Set2')
            )
            ax.set_title('Median Price/SqFt by Property Type')
            plt.xticks(rotation=30, ha='right'); plt.tight_layout(); st.pyplot(fig); plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df['Size_in_SqFt'], df['Price_in_Lakhs'], alpha=0.2, color='teal', s=5)
            ax.set_title('Size vs Price')
            ax.set_xlabel('Size (SqFt)'); ax.set_ylabel('Price (Lakhs)')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10).plot(
                kind='barh', ax=ax, color='steelblue'
            )
            ax.set_title('Top 10 Cities by Avg Price')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).head(10).plot(
                kind='barh', ax=ax, color='coral'
            )
            ax.set_title('Top 10 States by Avg Price/SqFt')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        if 'Good_Investment' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            df.groupby('City')['Good_Investment'].mean().sort_values(ascending=False).head(15).plot(
                kind='bar', ax=ax, color='green', alpha=0.8
            )
            ax.set_title('Good Investment Rate by City (Top 15)')
            ax.set_ylabel('Rate')
            plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        num_features = ['Price_in_Lakhs','Size_in_SqFt','Price_per_SqFt','BHK',
                        'Age_of_Property','Nearby_Schools','Nearby_Hospitals',
                        'Infrastructure_Score','Amenity_Count']
        existing = [c for c in num_features if c in df.columns]

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(df[existing].corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby('Amenity_Count')['Price_per_SqFt'].mean().plot(kind='bar', ax=ax, color='darkorange')
            ax.set_title('Amenity Count vs Avg Price/SqFt')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            order = [o for o in ['Low','Medium','High'] if o in df['Public_Transport_Accessibility'].values]
            sns.boxplot(data=df, x='Public_Transport_Accessibility', y='Price_per_SqFt',
                        palette='muted', order=order, ax=ax)
            ax.set_title('Transport Accessibility vs Price/SqFt')
            plt.tight_layout(); st.pyplot(fig); plt.close()

