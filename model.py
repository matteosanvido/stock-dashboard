import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_features(df):
    """
    Takes raw stock data and adds technical indicators
    that the model will use to make predictions.
    """

    # --- Moving Averages ---
    # Average closing price over last 5 and 20 days
    # Helps smooth out noise and spot trends
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # --- Daily Return ---
    # How much the stock moved today in % terms
    df['Daily_Return'] = df['Close'].pct_change()

    # --- Volatility ---
    # How wildly the stock has been swinging over the last 5 days
    df['Volatility'] = df['Daily_Return'].rolling(window=5).std()

    # --- Volume Change ---
    # Is trading volume higher or lower than usual?
    df['Volume_Change'] = df['Volume'].pct_change()

    # --- Target Label ---
    # 1 if tomorrow's price is higher than today's, 0 if lower
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with missing values (caused by rolling calculations)
    df.dropna(inplace=True)

    return df


def train_model(df):
    """
    Trains the Random Forest model on historical data
    and returns the trained model + its accuracy.
    """

    # Add our features to the data
    df = add_features(df)

    # Define which columns the model learns from
    features = ['MA5', 'MA20', 'Daily_Return', 'Volatility', 'Volume_Change']
    X = df[features]   # inputs
    y = df['Target']   # output (up or down)

    # Split into training data (80%) and testing data (20%)
    # shuffle=False is important — we don't want to mix past and future data
    # Make sure we have enough data to work with
    if len(X) < 50:
        return None, 0.0, df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Make sure test set isn't too small
    if len(X_test) < 10:
        return None, 0.0, df

    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test how accurate it is on data it has never seen
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, df


def predict_tomorrow(model, df):
    """
    Uses the trained model to predict whether the stock
    will go up or down tomorrow.
    """

    features = ['MA5', 'MA20', 'Daily_Return', 'Volatility', 'Volume_Change']

    # Take the most recent row of data (today)
    latest = df[features].iloc[-1].values.reshape(1, -1)

    # Get prediction and confidence
    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]

    return prediction, probability