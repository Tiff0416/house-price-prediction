# House Price Prediction

This project uses a Random Forest model to predict housing prices using data from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Project Structure

- `data_utils.py` — Downloads and unzips Kaggle competition data
- `model.py` — Trains a Random Forest model and generates evaluation plots
- `images/` — Contains prediction and feature importance plots
- `report.md` — Final written report of the project

## How to Run

1. Make sure you have a `kaggle.json` file and place it in `~/.kaggle/`
2. Run the following in terminal:

```bash
python data_utils.py
python model.py
```
