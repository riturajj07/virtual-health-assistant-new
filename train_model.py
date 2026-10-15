import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'symptom_disease.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SYMPTOMS = ['fever','cough','headache','body_pain','fatigue','sore_throat','runny_nose','nausea','vomiting','diarrhea','chills','shortness_of_breath']
PROFILES = {
    'Flu': ['fever','cough','body_pain','fatigue','chills','headache'],
    'Migraine': ['headache','nausea','vomiting','fatigue'],
    'Common Cold': ['cough','sore_throat','runny_nose','headache','fatigue'],
    'Typhoid': ['fever','headache','body_pain','fatigue','nausea','diarrhea'],
    'Viral Fever': ['fever','headache','body_pain','fatigue','chills'],
    'Gastroenteritis': ['nausea','vomiting','diarrhea','fatigue','fever','body_pain'],
    'Respiratory Infection': ['cough','fever','sore_throat','shortness_of_breath','fatigue'],
    'Allergy': ['cough','runny_nose','sore_throat','headache'],
    'Stress': ['headache','fatigue','nausea'],
    'Body Pain': ['body_pain','fatigue','headache'],
}

# Fixed seed examples from the original project, plus deterministic synthetic rows.
SEED_ROWS = [
    [1,1,0,1,0,0,0,0,0,0,0,0,'Flu'],
    [1,0,1,0,0,0,0,0,0,0,0,0,'Migraine'],
    [0,0,0,1,0,0,0,0,0,0,0,0,'Body Pain'],
    [1,1,1,1,0,0,0,0,0,0,0,0,'Typhoid'],
    [1,1,0,0,0,0,0,0,0,0,0,0,'Common Cold'],
    [0,0,1,0,0,0,0,0,0,0,0,0,'Stress'],
]

rng = np.random.default_rng(42)
rows = []
for row in SEED_ROWS:
    rows.append(dict(zip(SYMPTOMS + ['disease'], row)))
for disease, core in PROFILES.items():
    for _ in range(180):
        row = {s: 0 for s in SYMPTOMS}
        for s in core:
            row[s] = int(rng.random() < 0.78)
        for s in SYMPTOMS:
            if s not in core and rng.random() < 0.08:
                row[s] = 1
        if sum(row.values()) < 2:
            chosen = rng.choice(core, size=2, replace=False if len(core) > 1 else True)
            for s in chosen: row[s] = 1
        row['disease'] = disease
        rows.append(row)

df = pd.DataFrame(rows)[SYMPTOMS + ['disease']]
df.to_csv(DATA_PATH, index=False)
X = df[SYMPTOMS].astype(int)
encoder = LabelEncoder(); y = encoder.fit_transform(df['disease'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=350, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

with open(os.path.join(MODEL_DIR, 'health_model.pkl'), 'wb') as f: pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f: pickle.dump(encoder, f)
with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'wb') as f: pickle.dump(SYMPTOMS, f)

metrics = {
    'accuracy': round(float(accuracy_score(y_test, model.predict(X_test))) * 100, 2),
    'precision_weighted': round(float(precision_score(y_test, model.predict(X_test), average='weighted', zero_division=0)) * 100, 2),
    'recall_weighted': round(float(recall_score(y_test, model.predict(X_test), average='weighted', zero_division=0)) * 100, 2),
    'samples': int(len(df)), 'symptoms': len(SYMPTOMS), 'diseases': len(encoder.classes_)
}
with open(os.path.join(MODEL_DIR, 'metrics.pkl'), 'wb') as f: pickle.dump(metrics, f)

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.06, objective='multi:softprob', num_class=len(encoder.classes_), eval_metric='mlogloss', random_state=42, n_jobs=2)
    xgb.fit(X_train, y_train)
    with open(os.path.join(MODEL_DIR, 'xgboost_model.pkl'), 'wb') as f: pickle.dump(xgb, f)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
    metrics['xgboost_accuracy'] = round(float(xgb_acc), 2)
except Exception as e:
    metrics['xgboost_accuracy'] = None
    print('XGBoost skipped:', e)
with open(os.path.join(MODEL_DIR, 'metrics.pkl'), 'wb') as f: pickle.dump(metrics, f)

print('Training completed successfully.')
print(f"Dataset samples: {len(df)}")
print(f"Input symptoms ({len(SYMPTOMS)}): {', '.join(SYMPTOMS)}")
print(f"Output diseases ({len(encoder.classes_)}): {', '.join(encoder.classes_)}")
print('Random Forest test accuracy:', metrics['accuracy'], '%')
if metrics.get('xgboost_accuracy') is not None: print('XGBoost test accuracy:', metrics['xgboost_accuracy'], '%')
print('Note: synthetic samples are for academic software demonstration only, not medical training or diagnosis.')
