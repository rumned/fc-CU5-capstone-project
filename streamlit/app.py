import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='District crime rate predictor')


@st.cache_resource
def load_bundle():
    return joblib.load('crime_model.joblib')


bundle = load_bundle()
model = bundle['model']
features = bundle['features']
ranges = bundle['ranges']

pop_min = float(np.exp(ranges['log_pop']['min']))
pop_med = float(np.exp(ranges['log_pop']['median']))
pop_max = float(np.exp(ranges['log_pop']['max']))

st.title('District crime rate predictor')
st.write(
    'Enter a district profile to get the predicted assault and property crime '
    'rates per 1,000 people. The model is a random forest trained on 2022 and 2023 districts.'
)

left, right = st.columns(2)

with left:
    st.subheader('Inputs')
    population = st.number_input(
        'Population (thousands)', min_value=pop_min, max_value=pop_max,
        value=pop_med, step=10.0,
    )
    students_pc = st.number_input(
        'Students per 1,000 people',
        min_value=float(ranges['students_pc']['min']),
        max_value=float(ranges['students_pc']['max']),
        value=float(ranges['students_pc']['median']), step=1.0,
    )
    schools_pc = st.number_input(
        'Schools per 1,000 people',
        min_value=float(ranges['schools_pc']['min']),
        max_value=float(ranges['schools_pc']['max']),
        value=float(ranges['schools_pc']['median']), step=0.01,
    )
    income_median = st.number_input(
        'Median income',
        min_value=float(ranges['income_median']['min']),
        max_value=float(ranges['income_median']['max']),
        value=float(ranges['income_median']['median']), step=100.0,
    )

row = pd.DataFrame([{
    'log_pop': np.log(population),
    'students_pc': students_pc,
    'schools_pc': schools_pc,
    'income_median': income_median,
}])[features]

assault, prop = model.predict(row)[0]

with right:
    st.subheader('Predicted rates')
    st.metric('Assault per 1,000', f'{assault:.3f}')
    st.metric('Property crime per 1,000', f'{prop:.3f}')

st.caption(
    'Input limits are the smallest and largest values in the training data, '
    'because a random forest cannot predict outside the range it was trained on.'
)