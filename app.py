from flask import Flask, request, jsonify
import os, pickle, sqlite3
from datetime import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'feedback.db')
app = Flask(__name__)


def load_pickle(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing model file: {name}. Run train_model.py first.')
    with open(path, 'rb') as f: return pickle.load(f)

model = load_pickle('health_model.pkl')
encoder = load_pickle('label_encoder.pkl')
feature_columns = load_pickle('feature_columns.pkl')
try: xgb_model = load_pickle('xgboost_model.pkl')
except Exception: xgb_model = None
try: metrics = load_pickle('metrics.pkl')
except Exception: metrics = {}

ADVICE = {
 'Flu': ('Rest, drink plenty of fluids, and monitor your symptoms.', 'Seek medical care if symptoms are severe, worsening, or breathing becomes difficult.'),
 'Migraine': ('Rest in a quiet, dark environment and stay hydrated.', 'Seek urgent care for a sudden, unusually severe headache or new neurological symptoms.'),
 'Common Cold': ('Rest, hydrate, and monitor your symptoms.', 'Seek care if breathing becomes difficult, fever is persistent, or symptoms worsen.'),
 'Typhoid': ('A healthcare professional should evaluate persistent fever and gastrointestinal symptoms.', 'Prompt medical evaluation is recommended for persistent/high fever or worsening symptoms.'),
 'Viral Fever': ('Rest, hydrate, and monitor temperature and symptoms.', 'Seek care for persistent high fever, confusion, severe weakness, or worsening symptoms.'),
 'Gastroenteritis': ('Take fluids frequently and watch for dehydration.', 'Seek care for severe dehydration, blood in stool, severe pain, or persistent vomiting.'),
 'Respiratory Infection': ('Rest and monitor fever, cough, and breathing.', 'Seek urgent care for significant breathing difficulty, blue lips, or severe chest symptoms.'),
 'Allergy': ('Avoid suspected triggers and monitor symptoms.', 'Seek urgent care if facial/throat swelling or breathing difficulty develops.'),
 'Stress': ('Prioritize rest, sleep, hydration, and supportive conversations.', 'Seek professional support if symptoms are severe, persistent, or affecting daily functioning.'),
 'Body Pain': ('Rest, hydrate, and monitor the underlying symptoms.', 'Seek care if pain is severe, persistent, unexplained, or accompanied by concerning symptoms.')
}

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, prediction TEXT, rating TEXT, comment TEXT, created_at TEXT)')
init_db()

@app.route('/')
def home():
    return jsonify({'message':'Virtual Health Assistant API is running','features':feature_columns,'disease_count':len(encoder.classes_)})

@app.route('/health')
def health():
    return jsonify({'status':'ok','model_loaded':True,'xgboost_loaded':xgb_model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        raw = data.get('symptoms', data.get('features'))
        if raw is None: return jsonify({'error':'No symptoms were provided.'}), 400
        if isinstance(raw, dict): features = [int(bool(raw.get(s, 0))) for s in feature_columns]
        elif isinstance(raw, list): features = [int(bool(v)) for v in raw]
        else: return jsonify({'error':'Symptoms must be a list or object.'}), 400
        if len(features) != len(feature_columns): return jsonify({'error':f'Expected {len(feature_columns)} symptom values, received {len(features)}.'}), 400
        if not any(features): return jsonify({'error':'Please select at least one symptom.'}), 400
        X = np.asarray(features, dtype=float).reshape(1, -1)
        probs = model.predict_proba(X)[0]
        order = np.argsort(probs)[::-1]
        pred_idx = int(order[0]); disease = str(encoder.inverse_transform([pred_idx])[0])
        ranked = [{'disease':str(encoder.inverse_transform([int(i)])[0]),'probability':round(float(probs[i])*100,2)} for i in order[:5]]
        importances = getattr(model, 'feature_importances_', np.zeros(len(feature_columns)))
        contributions = [{'symptom':s.replace('_',' ').title(),'importance':round(float(imp),4)} for s,v,imp in zip(feature_columns,features,importances) if v]
        contributions.sort(key=lambda z:z['importance'], reverse=True)
        selected = [s.replace('_',' ').title() for s,v in zip(feature_columns,features) if v]
        explanation = 'The prediction is influenced most by: ' + ', '.join(x['symptom'] for x in contributions[:5]) + '.'
        advice, urgent = ADVICE.get(disease, ('Monitor symptoms and consult a qualified healthcare professional.','Seek professional care if symptoms are severe or worsening.'))
        shap_values = None
        shap_available = False
        try:
            import shap
            sv = shap.TreeExplainer(model)(X).values
            shap_values = sv[0].tolist() if np.ndim(sv) == 2 else np.asarray(sv).reshape(-1).tolist()
            shap_available = True
        except Exception: pass
        lime_available = False
        try:
            import lime  # noqa: F401
            lime_available = True
        except Exception: pass
        comparison = {'Random Forest': round(float(probs[pred_idx])*100,2)}
        if xgb_model is not None:
            try:
                xp = xgb_model.predict_proba(X)[0]; comparison['XGBoost'] = round(float(np.max(xp))*100,2)
            except Exception: pass
        return jsonify({'prediction':disease,'probability':ranked[0]['probability'],'top_predictions':ranked,'contributions':contributions,'explanation':explanation,'advice':advice,'urgent_advice':urgent,'selected_symptoms':selected,'model_comparison':comparison,'shap_available':shap_available,'lime_available':lime_available,'shap_values':shap_values,'evaluation_metrics':metrics})
    except Exception as e:
        app.logger.exception('Prediction error')
        return jsonify({'error':f'Prediction failed: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json(silent=True) or {}
        rating = str(data.get('rating','')).strip()
        if rating not in {'Helpful 👍','Somewhat Helpful 🙂','Not Helpful 👎'}: return jsonify({'error':'Invalid feedback rating.'}), 400
        with sqlite3.connect(DB_PATH) as con:
            con.execute('INSERT INTO feedback(prediction,rating,comment,created_at) VALUES(?,?,?,?)',(str(data.get('prediction','')),rating,str(data.get('comment',''))[:1000],datetime.now().isoformat(timespec='seconds')))
        return jsonify({'message':'Feedback saved successfully.'})
    except Exception as e: return jsonify({'error':str(e)}), 500

@app.route('/feedback/summary')
def feedback_summary():
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating').fetchall()
    return jsonify({'summary':{r[0]:r[1] for r in rows}})

if __name__ == '__main__': app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
