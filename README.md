# GETTING STARTED
Run flask HTTP server

>app.py

# API ENDPOINT
>'Content-Type: application/json'

```POST /predict```

## Request Body : 
    ```json
    {"sentence":<"Enter the list of Sentences">, "entity":<"enter the entities list">}
    ```
## Inference from the model: 
    ```json
    {"sentence":["the website was great", "the movie was good but I didn't like it"], "entity":["website","movie"]}
    ```
    ```json
    [{'Entity': 'website',
      'Sentence': 'the website was great',
      'pred_prob': array([-0.6650544, -0.722052 ], dtype=float32),
      'sentiment': 'Negative'},
     {'Entity': 'movie',
      'Sentence': "the movie was good but I didn't like it",
      'pred_prob': array([-0.6282205 , -0.76258385], dtype=float32),
      'sentiment': 'Negative'}]  
  ```


***
***
