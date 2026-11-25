"""Local portfolio demonstration. Use synthetic claims only."""
import os
import tempfile
import uuid
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
from insurance_agent import build_claims_graph, build_vectorstore_from_files, adjudicate

st.set_page_config(page_title='ClaimGuard AI', page_icon='🛡️', layout='wide')
st.title('ClaimGuard AI')
st.caption('Agentic RAG · Evidence-backed auto-insurance claims decision support')
st.warning('Portfolio prototype. Synthetic data only. Not an insurance decision system. Model providers receive submitted claim text and retrieved excerpts.')
with st.sidebar:
    st.header('Configuration')
    groq = st.text_input('Groq API key', type='password') or os.getenv('GROQ_API_KEY')
    tavily = st.text_input('Tavily API key (optional)', type='password') or os.getenv('TAVILY_API_KEY')
    web = st.checkbox('Search public regulatory sources (optional)', value=False, disabled=not bool(tavily))
    relevance = st.slider('Relevance threshold', 0.7, 1.0, 0.7, 0.05)
    grounding = st.slider('Grounding threshold', 0.7, 1.0, 0.7, 0.05)
    retries = st.slider('Retrieval retries', 0, 2, 2)
    uploads = st.file_uploader('Fictional policy files (optional)', type=['txt','pdf'], accept_multiple_files=True)
    if st.button('Build policy index'):
        try:
            with st.spinner('Embedding policy clauses…'), tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, f in enumerate(uploads):
                    path = Path(tmp) / f'policy_{i}{Path(f.name).suffix.lower()}'
                    path.write_bytes(f.getvalue()); paths.append(str(path))
                if not paths:
                    paths = [str(p) for p in sorted((Path(__file__).parent/'data'/'policies').glob('*.txt'))]
                st.session_state.index = build_vectorstore_from_files(paths)
            st.success('Policy index ready')
        except Exception:
            st.error('Indexing failed. Check file format and embedding model availability.')
with st.form('claim'):
    st.subheader('Claim details')
    a,b = st.columns(2)
    with a:
        kind=st.selectbox('Claim type',['hail','collision','theft','vandalism','glass_damage','animal_collision','mechanical_breakdown','other'])
        status=st.selectbox('Policy status',['active','expired','cancelled','unknown'])
        state=st.text_input('State','MO')
        use=st.selectbox('Vehicle use',['personal','rideshare','commercial_delivery','unknown'])
    with b:
        driver=st.selectbox('Driver status',['named_insured','listed_driver','excluded_driver','unknown'])
        incident=st.date_input('Incident date',date(2026,7,1))
        amount=st.number_input('Claim amount ($)',min_value=0.,value=1800.)
        evidence=st.text_input('Evidence (comma-separated)','Damage photos, inspection report')
    description=st.text_area('Synthetic claim description','Hail damaged the insured vehicle while parked during a storm.')
    submit=st.form_submit_button('Evaluate claim',type='primary')
if submit:
    if not groq or 'index' not in st.session_state:
        st.error('Configure a Groq key and build the policy index first.')
    else:
        try:
            client=None
            if web and tavily:
                from tavily import TavilyClient
                client=TavilyClient(api_key=tavily)
            with st.spinner('Retrieving, grading and verifying evidence…'):
                result=adjudicate(build_claims_graph(groq),uuid.uuid4().hex[:12],description,st.session_state.index,client,relevance,grounding,retries,web,claim_type=kind,policy_status=status,state=state,vehicle_use=use,driver_status=driver,incident_date=str(incident),claim_amount=amount,evidence=[x.strip() for x in evidence.split(',') if x.strip()])
            st.session_state.last=result['decision'].model_dump()
            st.session_state.audit=result['audit_record']
        except Exception:
            st.error('Evaluation failed. Route this claim to manual review; check provider configuration locally.')
if 'last' in st.session_state:
    d=st.session_state.last
    st.subheader('Latest recommendation: '+d['verdict'].upper())
    st.metric('Model-reported confidence (not calibrated)',f"{d['confidence']:.0%}")
    st.write(d['coverage_summary']); st.write(d['reasoning']); st.write(d['recommended_action'])
    st.write('Cited clauses:', ', '.join(d['cited_clause_ids']) or 'None')
    with st.expander('Audit record'):
        st.json(st.session_state.audit)
