import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from model import train_model, predict_tomorrow
from sentiment import get_headlines, analyze_sentiment, get_overall_sentiment

# --- Page Config ---
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Analysis Dashboard")

# --- User Input ---
ticker = st.text_input("Enter a stock ticker (e.g. AAPL, TSLA, MSFT):", value="AAPL")
period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y", "2y"])

# --- Fetch Data ---
if ticker:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    # Get company name from yfinance for better news search
    company_name = stock.info.get('longName', ticker)

    # --- Price Chart ---
    st.subheader(f"{ticker.upper()} — {company_name}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # --- Key Stats ---
    st.subheader("Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    col2.metric("High (Period)", f"${df['High'].max():.2f}")
    col3.metric("Low (Period)", f"${df['Low'].min():.2f}")
    col4.metric("Avg Volume", f"{int(df['Volume'].mean()):,}")

    # --- ML Prediction ---
    st.subheader("🤖 ML Prediction for Tomorrow")
    with st.spinner("Training model on historical data..."):
        model, accuracy, df_with_features = train_model(df.copy())

    if model is None:
        st.warning("⚠️ Not enough data to train the model. Try selecting a longer time period like 1y or 2y.")
    else:
        prediction, probability = predict_tomorrow(model, df_with_features)

        if prediction == 1:
            st.success("📈 Prediction: **UP** tomorrow")
        else:
            st.error("📉 Prediction: **DOWN** tomorrow")

        col1, col2 = st.columns(2)
        col1.metric("Model Confidence", f"{max(probability) * 100:.1f}%")
        col2.metric("Model Accuracy (on test data)", f"{accuracy * 100:.1f}%")
        st.caption("⚠️ This is for educational purposes only — not financial advice!")

    # --- Sentiment Analysis ---
    st.subheader("📰 News Sentiment Analysis")

    with st.spinner("Fetching latest news..."):
        articles = get_headlines(ticker, company_name)
        sentiment_df = analyze_sentiment(articles)
        avg_polarity, overall_sentiment = get_overall_sentiment(sentiment_df)

    if sentiment_df.empty:
        st.warning("No recent news found for this ticker.")
    else:
        # Overall sentiment score
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Sentiment", overall_sentiment)
        col2.metric("Avg Polarity Score", avg_polarity,
                    help="-1.0 = very negative, +1.0 = very positive")
        col3.metric("Articles Analyzed", len(sentiment_df))

        # Polarity bar chart
        fig2 = px.bar(
            sentiment_df,
            x='Date',
            y='Polarity',
            color='Polarity',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Headline Sentiment Over the Past 7 Days",
            hover_data=['Headline']
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

        # Headlines table
        with st.expander("View All Headlines"):
            st.dataframe(
                sentiment_df[['Date', 'Headline', 'Sentiment', 'Polarity']],
                use_container_width=True
            )

    # --- Raw Data ---
    with st.expander("View Raw Price Data"):
        st.dataframe(df.tail(20))