# Virtual Health Assistant with XAI Feedback

Enhanced version of the existing Flask + Streamlit Virtual Health Assistant.

## Features
- 12 symptom inputs
- 10 demonstration disease outputs
- Random Forest + XGBoost
- SHAP with safe fallback
- LIME dependency detection
- Top-5 probabilities
- Patient-friendly guidance and red-flag advice
- Patient feedback stored in SQLite
- Evaluation metrics
- Sample symptom patterns
- Attractive Streamlit dashboard

## Run on Windows
```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python train_model.py
```

Terminal 1:
```cmd
python app.py
```

Terminal 2:
```cmd
cd "C:\path\to\Virtual_Health_Assistant_XAI_Original_Updated"
venv\Scripts\activate
streamlit run streamlit_app.py
```

Open http://localhost:8501

## Important
The expanded dataset contains synthetic demonstration samples created for an academic software demo. It must not be presented as a clinically validated dataset or used for real diagnosis.
