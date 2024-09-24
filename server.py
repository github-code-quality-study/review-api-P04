import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any, Dict, List
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.valid_locations = [
            "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
            "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
            "El Paso, Texas", "Escondido, California", "Fresno, California",
            "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
            "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
            "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
        ]
    
    # Polarity/sentiment of review content:
    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores
    
    # Filter reviews based on location and time-stamp 
    def filter_reviews(self, location: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        filtered_reviews = reviews

        # If Location is provided 
        if location and location in self.valid_locations:
            filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

        # If there is a start date 
        if start_date:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S') >= start_date_dt]
        
        # If end-date is provided :
        if end_date:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], '%Y-%m-%d %H:%M:%S') <= end_date_dt]
        
        return filtered_reviews
    
    # GET:
    def get_response_body(self, reviews: List[Dict[str, Any]]) -> bytes:
        for review in reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

        reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

        return json.dumps(reviews, indent=2).encode("utf-8")

    def handle_get(self, query: Dict[str, List[str]], start_response: Callable[..., Any]) -> bytes:
        location = query.get("location", [None])[0]
        start_date = query.get("start_date", [None])[0]
        end_date = query.get("end_date", [None])[0]

        filtered_reviews = self.filter_reviews(location, start_date, end_date)
        response_body = self.get_response_body(filtered_reviews)

        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    def handle_post(self, environ: dict, start_response: Callable[..., Any]) -> bytes:
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            request_body = environ['wsgi.input'].read(content_length).decode("utf-8")
            post_data = parse_qs(request_body)

            location = post_data.get("Location", [None])[0]
            review_body = post_data.get("ReviewBody", [None])[0]

            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'Missing Location or ReviewBody']

            if location not in self.valid_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'Invalid Location']

            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "Location": location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ReviewBody": review_body,
                "sentiment": self.analyze_sentiment(review_body)
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json")])
            return [str(e).encode("utf-8")]
        
    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        
        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            query = parse_qs(environ.get('QUERY_STRING', ''))
            return self.handle_get(query, start_response)
            
        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            return self.handle_post(environ, start_response)

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()