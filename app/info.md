The original dataset files shall be placed to the dir [data](data/).

All the experiments were performed based on the dataset files parsed by chunks. To launch parsing the following modes are used:

- "parse_visits"
- "parse_orders"
- "parse_meta"
- "parse_accepted"

Example:

```
python main.py -m parse_visits
```

These modes are used to perform experiments with catboost and logreg classifiers:

"catboost_fit_visits", "catboost_fit_orders", "catboost_fit_orders_scores", "catboost_fit_meta", 

"logreg_fit_meta", "logreg_fit_both", "logreg_fit_brands",

The final prediction is performed by with logreg:

```
python main.py -m logreg_predict
```