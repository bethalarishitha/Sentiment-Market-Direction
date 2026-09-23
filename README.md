# Investor-Sentiment-Driven Market Direction Prediction

## Quick start

```powershell
pip install -r requirements.txt
python src/smd_data_collection.py   # Sprint 1.1: writes CSVs + manifest.json to smd_data/raw_data/
python src/smd_preprocessing.py     # Sprint 1.2: clean features + EDA charts (reports/eda/)
python src/smd_modeling.py          # Sprint 2.1: first models + score cards (reports/models/)
python src/smd_model_comparison.py  # Sprint 2.2: full grid + HTML overview (reports/comparison/)
python src/smd_validation.py        # Sprint 4: validate best models on QQQ + side-by-side charts
python src/smd_live_predict.py      # Next-day / next-week UP-DOWN (S&P focus)
python src/smd_dynamic_predict.py   # Answer tab: all dataset stocks (GSPC, QQQ, basket)
python src/smd_dashboard.py         # Tabs + dynamic ticker train/predict (http://127.0.0.1:8050)
```

## Tools - Explanation

These are the main tools for the project and what each one actually does for me.


| Tool                  | What it is and why I use it                                                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **yfinance**          | A free library that downloads stock and index prices from Yahoo Finance. It gives me the S&P 500, QQQ, VIX and the basket of big companies.               |
| **curl_cffi session** | A helper that makes my download look like a normal web browser. Yahoo blocks robots that ask for too much, so this keeps my downloads from being blocked. |
| **VADER**             | A simple tool that reads a news headline and scores it as good news or bad news. I use it to turn text into a mood number.                                |
| **scikit-learn**      | The main machine learning toolbox. I use it for the first models, Logistic Regression and Random Forest, and for the score cards like accuracy and F1.    |
| **LightGBM**          | A fast and powerful model that finds patterns in tables of numbers. It becomes my strongest classic model in a later sprint.                              |
| **GRU (PyTorch)**     | A small deep learning model that learns from the order of days over time, so it can spot patterns across a run of days rather than one day alone.         |
| **SMOTE**             | A trick that balances the Up days and Down days so the model cannot cheat by always guessing the more common answer.                                      |
| **Plotly Dash**       | The library I use to build the final dashboard, a small web page where the results and charts can be explored with the mouse.                             |


