
# app.py - Save this as stock_predictor_streamlit.py and run: streamlit run stock_predictor_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #00d4aa; text-align: center; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #00d4aa 0%, #00a085 100%); color: white; padding: 1rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Predict stock prices based on trading volume using Linear Regression!**")

# Sidebar for inputs
st.sidebar.header("📊 Prediction Inputs")
volume = st.sidebar.number_input("Enter Trading Volume (millions)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)

# Load or train model
@st.cache_resource
def load_or_train_model():
    # Sample data adapted for stock prediction
    data = {
        'volume': [1.79, 2.03, 2.5, 3.2],
        'price': [114.30, 114.20, 150.00, 220.00]
    }
    df = pd.DataFrame(data)
    
    # Train model
    model = LinearRegression()
    model.fit(df[['volume']], df['price'])
    
    # Save coefficients for display
    st.session_state.model = model
    st.session_state.coef = model.coef_[0]
    st.session_state.intercept = model.intercept_
    
    return model, df

model, sample_df = load_or_train_model()

# Main prediction
if st.button("🔮 Predict Stock Price", type="primary"):
    pred_price = model.predict([[volume]])[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Price", f"${pred_price:,.2f}", f"${pred_price:,.2f}")
    with col2:
        st.metric("Volume", f"{volume:.2f}M shares")
    with col3:
        st.metric("Price per M volume", f"${model.coef_[0]:.2f}")
    
    st.success(f"📈 For **{volume:.2f}M volume**, predicted price is **${pred_price:,.2f}**!")

# Model info
with st.expander("📊 Model Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Slope (coef)", f"${model.coef_[0]:.2f}/M volume")
        st.metric("Intercept", f"${model.intercept_:,.2f}")
    with col2:
        st.write("**Equation:** `price = {:.2f} × volume + {:.2f}`".format(
            model.coef_[0], model.intercept_))
        st.info("✅ Model trained on stock volume dataset!")

# Visualization
st.header("📈 Visualizations")

col1, col2 = st.columns(2)

with col1:
    # Scatter plot with prediction line
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample_df['volume'], sample_df['price'], color='orange', s=100, label='Training Data')
    
    # Prediction line
    x_range = np.linspace(sample_df['volume'].min()-0.5, sample_df['volume'].max()+0.5, 100)
    y_line = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_line, color='#00d4aa', linewidth=2, label='Regression Line')
    
    # Prediction point
    ax.scatter([volume], [model.predict([[volume]])[0]], color='red', s=150, 
               marker='*', label=f'Prediction ({volume:.2f}M)')
    
    ax.set_xlabel('Trading Volume (millions)')
    ax.set_ylabel('Stock Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    # Interactive price vs volume
    fig = px.scatter(sample_df, x='volume', y='price', 
                     title="Training Data Points",
                     labels={'volume': 'Trading Volume (millions)', 'price': 'Stock Price ($)'},
                     size_max=20)
    fig.add_hline(y=model.intercept_, line_dash="dash", 
                  annotation_text="Intercept")
    st.plotly_chart(fig, use_container_width=True)

# Batch prediction
st.header("📋 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV with 'volume' column", type=['csv'])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    if 'volume' in test_df.columns:
        test_df['predicted_price'] = model.predict(test_df[['volume']])
        test_df['price_per_volume'] = test_df['predicted_price'] / test_df['volume']
        
        st.dataframe(test_df.style.highlight_max(axis=0), use_container_width=True)
        
        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Predictions",
            data=csv,
            file_name='stock_price_predictions.csv',
            mime='text/csv'
        )
    else:
        st.error("❌ CSV must have 'volume' column!")

# Save model (from your notebook)
if st.button("💾 Save Model (Joblib)"):
    joblib.dump(model, "stock_price_model.joblib")
    st.success("✅ Model saved as 'stock_price_model.joblib'!")

if st.button("📥 Load Model (if saved)"):
    try:
        loaded_model = joblib.load("stock_price_model.joblib")
        st.success("✅ Model loaded successfully!")
        st.info(f"Loaded coef: ${loaded_model.coef_[0]:.2f}")
    except:
        st.warning("⚠️ No saved model found. Using fresh model.")

# Footer
st.markdown("---")
st.markdown("*Built for stock market analysis • Powered by Streamlit & scikit-learn*")
