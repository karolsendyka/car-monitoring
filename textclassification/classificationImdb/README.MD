Project which doesn classification of movie description to categories "positive" and  "negative"

Follow steps 


# Run
To build model use [build-model.py](build-model.py)
Before you run it yuou need to get indb dataset
```
!wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xzf aclImdb_v1.tar.gz
```


To use model use cli.py eg ```python cli.py ./model.keras ./tokenizer_config.json "Movie very strannge movie and interesting a bit"```
