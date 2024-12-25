import tweepy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import dash
from dash import dcc, html

API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
ACCESS_TOKEN = 'your_access_token'
ACCESS_SECRET = 'your_access_secret'

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)


# Fetch Tweets
def fetch_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en", tweet_mode="extended").items(count)
    data = [{"text": tweet.text, "created_at": tweet.created_at, "user": tweet.user.screen_name} for tweet in tweets]
    return data


tweets = fetch_tweets("radicalization", "#StopTheSteal", "Elon Musk 2024", "President Musk", count=200)
print(tweets)

for tweet in tweets:
    print(tweet.full_text)

analyzer = SentimentIntensityAnalyzer()


score = analyzer.polarity_scores(tweets)
print(score)

def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"


sentiments = [analyze_sentiment(tweet['text']) for tweet in tweets]
print(sentiments)

# Preprocess tweets
corpus = [tweet['text'] for tweet in tweets]
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Apply LDA (Latent Dirichlet Allocation)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Display topics
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])

# Prepare data
df = pd.DataFrame({
    "date": [tweet["created_at"] for tweet in tweets],
    "sentiment": sentiments
})

app = dash.Dash(__name__)

# Sentiment plot
fig = px.line(df, x="date", y="sentiment", title="Sentiment Over Time")
fig.show()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run_server(debug=True)