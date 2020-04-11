# GETTING STARTED
Run flask HTTP server

>app.py

# API ENDPOINT
>'Content-Type: application/json'

```POST /predict```

## Request Body : 
    ```json
    {"sentence":<"Enter your Sentence">, "entity":<"enter the entity">}
    ```
## Response from the model: 
    ```json
      {'Entity': 'movies',
      'Sentence': 'the website was great',
      'pred_prob': array([-5.093779  , -0.00615368], dtype=float32),
      'sentiment': 'Positive'},
     {'Entity': 'movies',
      'Sentence': 'the movie was great',
      'pred_prob': array([-5.419831e+00, -4.437718e-03], dtype=float32),
      'sentiment': 'Positive'}   
  ```


***
***
