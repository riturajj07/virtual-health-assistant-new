import os, pickle, requests
import streamlit as st

st.set_page_config(page_title='Virtual Health Assistant', page_icon='🩺', layout='wide', initial_sidebar_state='expanded')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
API = 'http://127.0.0.1:5000'

@st.cache_data
def load_data():
    with open(os.path.join(MODEL_DIR,'feature_columns.pkl'),'rb') as f: symptoms=pickle.load(f)
    try:
        with open(os.path.join(MODEL_DIR,'metrics.pkl'),'rb') as f: metrics=pickle.load(f)
    except Exception: metrics={}
    return symptoms, metrics

symptoms, metrics = load_data()

def reset():
    st.session_state.pop('result', None)
    st.session_state['feedback_sent'] = False

st.markdown('''<style>
.block-container{padding-top:1.2rem;max-width:1200px}.hero{padding:30px;border-radius:24px;background:linear-gradient(135deg,#0f766e,#2563eb);color:white;box-shadow:0 12px 30px rgba(37,99,235,.18);margin-bottom:22px}.hero h1{font-size:40px;margin:0}.hero p{font-size:17px;margin:8px 0 0}.pill{display:inline-block;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.16);margin-top:12px}.card{padding:20px;border-radius:18px;border:1px solid #e6eaf0;background:#fff;margin:10px 0}.section{font-size:24px;font-weight:700;margin-top:12px}.small{color:#64748b}.stButton>button{border-radius:12px;font-weight:600}.result{padding:22px;border-radius:18px;border:1px solid #dbeafe;background:#eff6ff}.warning{padding:16px;border-radius:14px;background:#fff7ed;border:1px solid #fed7aa}.sample{padding:14px;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0}
</style>''', unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🩺 Virtual Health Assistant</h1><p>AI-assisted symptom screening with Explainable AI feedback</p><span class="pill">Educational decision-support project • Not a medical diagnosis</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('🧭 Project Workflow')
    st.write('1. Select symptoms')
    st.write('2. Analyze with ML')
    st.write('3. Understand the result')
    st.write('4. Read patient guidance')
    st.write('5. Give feedback')
    st.divider()
    st.metric('Input symptoms', len(symptoms))
    st.metric('Output conditions', metrics.get('diseases',10))
    if st.button('🔄 New Assessment', use_container_width=True): reset(); st.rerun()

st.markdown('<div class="section">1. How are you feeling?</div>', unsafe_allow_html=True)
st.caption('Select all symptoms that apply. The more relevant symptoms you provide, the more informative the demonstration output can be.')
selected=[]
cols=st.columns(3)
for i,s in enumerate(symptoms):
    if cols[i%3].checkbox(s.replace('_',' ').title(), key='input_'+s): selected.append(s)
st.write(f'**{len(selected)} symptoms selected**')

with st.expander('🧪 Try sample inputs'):
    samples={'Flu-like pattern':['fever','cough','body_pain','fatigue','chills'],'Migraine-like pattern':['headache','nausea','vomiting'],'Cold-like pattern':['cough','sore_throat','runny_nose'],'Digestive pattern':['nausea','vomiting','diarrhea','fever']}
    for name, vals in samples.items(): st.markdown(f'<div class="sample"><b>{name}</b><br>{", ".join(x.replace("_"," ").title() for x in vals)}</div>',unsafe_allow_html=True)

if st.button('🔍 Analyze Symptoms', type='primary', use_container_width=True):
    if not selected: st.warning('Please select at least one symptom.')
    else:
        values=[1 if s in selected else 0 for s in symptoms]
        try:
            r=requests.post(f'{API}/predict',json={'symptoms':values},timeout=30)
            result=r.json()
            if not r.ok: st.error(result.get('error','Prediction failed.'))
            else: st.session_state['result']=result; st.session_state['feedback_sent']=False
        except requests.exceptions.ConnectionError:
            st.error('Could not connect to Flask API. Keep the backend terminal running with: python app.py')
        except requests.exceptions.Timeout: st.error('The API took too long to respond. Please try again.')
        except Exception as e: st.error(f'Unable to process request: {e}')

result=st.session_state.get('result')
if result:
    st.divider(); st.markdown('<div class="section">2. Your preliminary result</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric('Possible condition',result['prediction']); c2.metric('Model confidence',f"{result.get('probability',0):.1f}%"); c3.metric('Symptoms considered',len(result.get('selected_symptoms',[])))
    st.markdown(f'<div class="result"><h2>🩺 {result["prediction"]}</h2><p>{result.get("explanation","")}</p></div>',unsafe_allow_html=True)
    st.info('This is a preliminary software-generated result for academic demonstration. It is not a confirmed diagnosis.')

    st.markdown('<div class="section">3. Explainable AI — Why?</div>',unsafe_allow_html=True)
    st.write(result.get('explanation',''))
    contrib=result.get('contributions',[])
    if contrib: st.bar_chart({x['symptom']:x['importance'] for x in contrib})
    st.caption(f"SHAP available: {'Yes' if result.get('shap_available') else 'Fallback explanation'} • LIME installed: {'Yes' if result.get('lime_available') else 'No'}")

    st.markdown('<div class="section">4. Patient guidance</div>',unsafe_allow_html=True)
    st.success('💡 '+result.get('advice','Monitor your symptoms and consult a healthcare professional.'))
    st.warning('🚨 '+result.get('urgent_advice','Seek professional care if symptoms are severe or worsening.'))

    with st.expander('View selected symptoms'): st.write(', '.join(result.get('selected_symptoms',[])))
    st.markdown('<div class="section">5. Top possible outputs</div>',unsafe_allow_html=True)
    top=result.get('top_predictions',[])
    if top: st.dataframe(top,use_container_width=True,hide_index=True)

    st.markdown('<div class="section">6. Model comparison & evaluation</div>',unsafe_allow_html=True)
    if result.get('model_comparison'): st.bar_chart(result['model_comparison'])
    em=result.get('evaluation_metrics',{})
    if em:
        a,b,c=st.columns(3); a.metric('RF test accuracy',f"{em.get('accuracy',0):.1f}%"); b.metric('Precision',f"{em.get('precision_weighted',0):.1f}%"); c.metric('Recall',f"{em.get('recall_weighted',0):.1f}%")
        st.caption('Metrics are calculated on the demonstration dataset and should not be interpreted as clinical performance.')

    st.markdown('<div class="section">7. Patient feedback</div>',unsafe_allow_html=True)
    st.write('Your feedback helps evaluate whether the prediction and explanation were understandable.')
    rating=st.radio('Was this result and explanation helpful?',['Helpful 👍','Somewhat Helpful 🙂','Not Helpful 👎'],horizontal=True)
    comment=st.text_area('Optional comment',placeholder='Example: The explanation was easy to understand...')
    if st.button('💬 Submit Feedback',use_container_width=True):
        try:
            rr=requests.post(f'{API}/feedback',json={'prediction':result['prediction'],'rating':rating,'comment':comment},timeout=10)
            if rr.ok: st.session_state['feedback_sent']=True
            else: st.error(rr.json().get('error','Could not save feedback.'))
        except Exception as e: st.error(f'Could not save feedback: {e}')
    if st.session_state.get('feedback_sent'): st.success('❤️ Thank you! Your feedback has been recorded successfully.')

with st.expander('⚠️ Medical disclaimer'):
    st.write('This application is an educational project for preliminary decision support. It does not confirm disease, prescribe medicines, or replace a qualified healthcare professional. For severe, persistent, or concerning symptoms, seek professional medical care.')
