import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='District Crime Planner', page_icon='🧭', layout='wide')

CRIMES = {'assault': 'Assault', 'property': 'Property crime'}
PAGES = ['Overview', 'Districts to review', 'District profile', 'Profile check', 'About the data and models']
STATUSES = ['Review', 'Monitor', 'No flag']
STATUS_COLOURS = ['#b42318', '#c27c0e', '#5b7f74']
STATUS_CLASSES = {'Review': 'status-review', 'Monitor': 'status-monitor', 'No flag': 'status-noflag'}
LINE_COLOURS = ['#1d4e89', '#7a8ca3', '#c9a227']

# inputs of the cross-district model, with how they are shown in the app.
# shares are stored as fractions and shown as percentages
PROFILE_INPUTS = {
    'log_pop': ('Population (thousands)', 'population', 10.0, '%.1f'),
    'students_pc': ('Students per 1,000 people', 'plain', 5.0, '%.1f'),
    'schools_pc': ('Schools per 1,000 people', 'plain', 0.05, '%.2f'),
    'income_median': ('Median household income (RM a month)', 'plain', 100.0, '%.0f'),
    'income_mean': ('Mean household income (RM a month)', 'plain', 100.0, '%.0f'),
    'poverty_absolute': ('Absolute poverty (% of households)', 'plain', 0.5, '%.1f'),
    'poverty_relative': ('Relative poverty (% of households)', 'plain', 0.5, '%.1f'),
    'gini': ('Income inequality (gini, 0 to 1)', 'plain', 0.01, '%.3f'),
    'piped_water': ('Households with piped water (%)', 'plain', 1.0, '%.1f'),
    'young_male_share': ('Males aged 15–29 (% of population)', 'share', 0.5, '%.1f'),
    'age65_share': ('People aged 65 and over (% of population)', 'share', 0.5, '%.1f'),
    'male_share': ('Males (% of population)', 'share', 0.5, '%.1f'),
}

st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1 {font-weight: 700; letter-spacing: -0.01em;}
.page-lead {font-size: 1.05rem; color: #3b4859; max-width: 72ch; margin: -0.4rem 0 1.4rem 0; line-height: 1.55;}
.stat {border-left: 4px solid var(--accent); padding: 0.2rem 0 0.2rem 0.85rem; margin-bottom: 1rem; --accent: #1d4e89;}
.stat-review {--accent: #b42318;}
.stat-monitor {--accent: #c27c0e;}
.stat-noflag {--accent: #5b7f74;}
.stat-quiet {--accent: #c5ced9;}
.stat-label {font-size: 0.86rem; color: #556274;}
.stat-value {font-size: 1.65rem; font-weight: 650; color: #17212e; line-height: 1.25;}
.stat-note {font-size: 0.84rem; color: #556274;}
.status {display: inline-block; padding: 0.12rem 0.65rem; border-radius: 4px; font-size: 0.86rem; font-weight: 600; color: #ffffff;}
.status-review {background: #b42318;}
.status-monitor {background: #c27c0e;}
.status-noflag {background: #5b7f74;}
.district-heading {font-size: 1.05rem; color: #3b4859; margin: -0.6rem 0 1rem 0;}
.district-heading .status {margin-left: 0.4rem;}
.next-step {border: 1px solid #d6dde6; border-left: 6px solid var(--accent); border-radius: 4px; padding: 0.9rem 1.1rem; margin: 0.4rem 0 1rem 0; color: #17212e; --accent: #5b7f74;}
.next-step.stat-review {--accent: #b42318;}
.next-step.stat-monitor {--accent: #c27c0e;}
.next-step p {margin: 0.2rem 0 0.4rem 0; max-width: 75ch; line-height: 1.5;}
.small-note {font-size: 0.86rem; color: #556274; max-width: 75ch;}
</style>
""", unsafe_allow_html=True)


# ---------- loading ----------

@st.cache_resource
def load_bundle():
    return joblib.load('crime_model.joblib')


@st.cache_data
def load_tables():
    districts = pd.read_csv('app_districts.csv')
    years = pd.read_csv('app_district_years.csv')
    return districts, years


bundle = load_bundle()
districts, years = load_tables()
year_ahead = bundle['year_ahead']
cross_district = bundle['cross_district']


@st.cache_data
def outlook_2024(districts):
    # runs the year-ahead models on each district's 2024 profile
    features_2024 = districts[[feature + '_2024' for feature in year_ahead['features']]]
    features_2024.columns = year_ahead['features']
    return pd.DataFrame({crime: year_ahead['models'][f'{crime}_per_1000'].predict(features_2024) for crime in CRIMES},
                        index=districts['district'])


@st.cache_data
def cluster_names(districts):
    # the k-means clusters ordered by average population: smallest first
    order = districts.groupby('cluster')['log_pop'].mean().sort_values().index
    return dict(zip(order, ['Smaller districts', 'Mid-sized districts', 'Larger districts']))


@st.cache_data
def district_table(districts, years, crime, above_pct, rising_pct, min_cases):
    # one row per district with the rates, the expected rate, the change and the flags for one crime type
    rates = years[years['year'].between(2020, 2023)].pivot(index='district', columns='year', values=f'{crime}_per_1000')
    cases_2023 = years[years['year'] == 2023].set_index('district')[f'crimes_{crime}']
    table = districts[['district', 'state', 'cluster']].copy()
    table['average'] = districts[f'{crime}_avg']
    table['expected'] = districts[f'{crime}_expected']
    table['gap_pct'] = (table['average'] / table['expected'] - 1) * 100
    table['rate_2023'] = table['district'].map(rates[2023])
    table['earlier_average'] = table['district'].map(rates[[2020, 2021, 2022]].mean(axis=1))
    table['change_pct'] = (table['rate_2023'] / table['earlier_average'] - 1) * 100
    table['cases_2023'] = table['district'].map(cases_2023)
    table['outlook_2024'] = table['district'].map(outlook_2024(districts)[crime])
    table['above_expected'] = table['gap_pct'] >= above_pct
    table['rising'] = (table['change_pct'] >= rising_pct) & (table['cases_2023'] >= min_cases)
    table['status'] = np.select([table['above_expected'] & table['rising'], table['above_expected'] | table['rising']],
                                ['Review', 'Monitor'], 'No flag')
    return table


@st.cache_data
def rates_by_year(years, group):
    # crimes per 1,000 people per year, for the whole country (group=None) or for each state
    keys = ['year'] if group is None else [group, 'year']
    totals = years.groupby(keys)[['crimes_assault', 'crimes_property', 'population']].sum(min_count=1).reset_index()
    for crime in CRIMES:
        totals[f'{crime}_rate'] = totals[f'crimes_{crime}'] / totals['population']
    return totals


def stat(label, value, note='', kind=''):
    return (f'<div class="stat {kind}"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>')


def show_stats(items):
    for column, item in zip(st.columns(len(items)), items):
        column.markdown(stat(*item), unsafe_allow_html=True)


def status_badge(status):
    return f'<span class="status {STATUS_CLASSES[status]}">{status}</span>'


def signed(value, decimals=0, suffix='%'):
    return f'{value:+.{decimals}f}{suffix}'


def go_to(page, district=None):
    # used as a button callback, so the page changes before the next run
    st.session_state['page'] = page
    if district is not None:
        st.session_state['selected_district'] = district


def remember_district(widget_key):
    # keeps the chosen district when the user moves to another page
    st.session_state['selected_district'] = st.session_state[widget_key]


def district_label(name):
    return f"{name}, {districts.set_index('district').loc[name, 'state']}"


# ---------- sidebar ----------

if 'page' not in st.session_state:
    st.session_state['page'] = PAGES[0]
if 'selected_district' not in st.session_state:
    st.session_state['selected_district'] = sorted(districts['district'])[0]

with st.sidebar:
    st.markdown('### District Crime Planner')
    st.caption('For crime-prevention planners deciding which Malaysian districts need a closer look.')
    page = st.radio('Page', PAGES, key='page')
    crime = st.radio('Crime type', list(CRIMES), format_func=CRIMES.get, key='crime')
    with st.expander('Flag settings'):
        above_pct = st.slider('Above expected by at least (%)', 5, 100, 25, step=5)
        rising_pct = st.slider('2023 rate above the 2020–2022 average by at least (%)', 5, 100, 15, step=5)
        min_cases = st.number_input('Minimum cases in 2023 for a rising flag', 0, 200, 20, step=5)
    st.caption('Data: crime, population, household income, poverty, inequality, amenities, schools and enrolment '
               'by district, 2016–2024.')

table = district_table(districts, years, crime, above_pct, rising_pct, min_cases)
names = cluster_names(districts)
crime_name = CRIMES[crime]


# ---------- overview ----------

def overview_page():
    st.title('Crime across Malaysia’s districts')
    st.markdown(
        '<p class="page-lead">Crime rates depend a lot on what a district is like: larger, richer districts '
        'record more crime per person. This tool compares each district with the rate expected for districts '
        'with a similar profile, checks whether its crime is rising, and helps you decide where to look first.</p>',
        unsafe_allow_html=True)

    national = rates_by_year(years, None).set_index('year')
    rate_2023, rate_2022 = national.loc[2023, f'{crime}_rate'], national.loc[2022, f'{crime}_rate']
    cases_2023, cases_2016 = national.loc[2023, f'crimes_{crime}'], national.loc[2016, f'crimes_{crime}']
    counts = table['status'].value_counts()
    show_stats([
        (f'{crime_name} per 1,000 people, 2023', f'{rate_2023:.2f}', f'{signed((rate_2023 / rate_2022 - 1) * 100)} from 2022'),
        (f'{crime_name} cases, 2023', f'{cases_2023:,.0f}', f'{signed((cases_2023 / cases_2016 - 1) * 100)} from 2016'),
        ('Districts to review', counts.get('Review', 0), 'above expected and rising', 'stat-review'),
        ('Districts to monitor', counts.get('Monitor', 0), 'crime above expected', 'stat-monitor'),
    ])

    left, right = st.columns(2)
    with left:
        st.subheader('Cases per year')
        cases = national.reset_index()[['year', f'crimes_{crime}']].rename(columns={f'crimes_{crime}': 'cases'})
        chart = alt.Chart(cases).mark_line(point=True, color=LINE_COLOURS[0]).encode(
            x=alt.X('year:O', title=None), y=alt.Y('cases:Q', title='Cases'),
            tooltip=['year', alt.Tooltip('cases:Q', format=',')])
        st.altair_chart(chart)
    with right:
        st.subheader('Rate by state, 2023')
        states = rates_by_year(years, 'state')
        states = states[states['year'] == 2023][['state', f'{crime}_rate']].rename(columns={f'{crime}_rate': 'rate'})
        bars = alt.Chart(states).mark_bar(color=LINE_COLOURS[0]).encode(
            x=alt.X('rate:Q', title='Per 1,000 people'), y=alt.Y('state:N', sort='-x', title=None),
            tooltip=['state', alt.Tooltip('rate:Q', format='.2f')])
        national_line = alt.Chart(pd.DataFrame({'rate': [rate_2023]})).mark_rule(color='#c9a227', strokeDash=[4, 3]).encode(x='rate:Q')
        st.altair_chart(bars + national_line)
        st.caption('The dashed line is the national rate.')

    highest_state = states.sort_values('rate').iloc[-1]
    above = (table['gap_pct'] >= above_pct).sum()
    st.subheader('What this shows')
    st.markdown(
        f'- {crime_name} cases {"fell" if cases_2023 < cases_2016 else "rose"} from {cases_2016:,.0f} in 2016 to {cases_2023:,.0f} in 2023.\n'
        f'- {highest_state["state"]} has the highest rate in 2023 ({highest_state["rate"]:.2f} per 1,000 people).\n'
        f'- {above} of 131 districts recorded at least {above_pct}% more {crime_name.lower()} in 2020–2023 than '
        f'districts with a similar profile.')

    st.subheader('Next steps')
    first, second, _ = st.columns([1, 1, 2])
    first.button('Open the review list', on_click=go_to, args=('Districts to review',), type='primary')
    second.button('Check a district profile', on_click=go_to, args=('Profile check',))


# ---------- districts to review ----------

def review_page():
    st.title('Districts to review')
    st.markdown(
        f'<p class="page-lead">Districts whose {crime_name.lower()} rate is higher than expected for their profile, '
        'or rising, are flagged here. Use the list to choose districts for a closer look, then open a district '
        'profile to see why it was flagged.</p>', unsafe_allow_html=True)

    with st.expander('How the flags work'):
        st.markdown(
            f'- **Expected rate**: the 2020–2023 average rate that the cross-district model predicts for a district with '
            f'this population, income, poverty, education and age profile and state. Each district’s expected rate '
            f'comes from a model trained without that district.\n'
            f'- **Above expected**: the 2020–2023 average is at least {above_pct}% above the expected rate.\n'
            f'- **Rising**: the 2023 rate is at least {rising_pct}% above the district’s 2020–2022 average, with at '
            f'least {min_cases} cases in 2023, so that a few extra cases in a small district do not set the flag.\n'
            f'- **Review**: both flags. **Monitor**: one flag. The thresholds can be changed under Flag settings.')

    counts = table['status'].value_counts()
    show_stats([(status, counts.get(status, 0), f'of {len(table)} districts', STATUS_CLASSES[status].replace('status', 'stat'))
                for status in STATUSES])

    st.subheader('Actual compared with expected, 2020–2023')
    status_scale = alt.Scale(domain=STATUSES, range=STATUS_COLOURS)
    limit = float(max(table['average'].max(), table['expected'].max()) * 1.05)
    points = alt.Chart(table).mark_circle(size=70, opacity=0.85).encode(
        x=alt.X('expected:Q', title='Expected rate per 1,000 people', scale=alt.Scale(domain=[0, limit])),
        y=alt.Y('average:Q', title='Actual average rate per 1,000 people', scale=alt.Scale(domain=[0, limit])),
        color=alt.Color('status:N', scale=status_scale, title='Status'),
        tooltip=['district', 'state', 'status', alt.Tooltip('average:Q', format='.2f', title='actual'),
                 alt.Tooltip('expected:Q', format='.2f'), alt.Tooltip('gap_pct:Q', format='+.0f', title='gap (%)')])
    diagonal = alt.Chart(pd.DataFrame({'x': [0, limit], 'y': [0, limit]})).mark_line(color='#7a8ca3', strokeDash=[4, 3]).encode(x='x', y='y')
    st.altair_chart(diagonal + points)
    st.caption('Districts above the dashed line recorded more crime than expected for their profile.')

    st.subheader('List')
    filter_left, filter_right = st.columns(2)
    chosen_statuses = filter_left.multiselect('Status', STATUSES, default=['Review', 'Monitor'])
    chosen_states = filter_right.multiselect('State', sorted(table['state'].unique()), placeholder='All states')
    shown = table[table['status'].isin(chosen_statuses)]
    if chosen_states:
        shown = shown[shown['state'].isin(chosen_states)]
    shown = shown.assign(status_order=shown['status'].map({status: i for i, status in enumerate(STATUSES)}))
    shown = shown.sort_values(['status_order', 'gap_pct'], ascending=[True, False])
    columns = ['district', 'state', 'status', 'average', 'expected', 'gap_pct', 'rate_2023', 'change_pct', 'cases_2023', 'outlook_2024']

    if shown.empty:
        st.info('No districts match these filters. Add a status or a state, or lower the thresholds under Flag settings.')
        return
    st.dataframe(shown[columns], hide_index=True, column_config={
        'district': 'District', 'state': 'State', 'status': 'Status',
        'average': st.column_config.NumberColumn('2020–2023 average', format='%.2f'),
        'expected': st.column_config.NumberColumn('Expected', format='%.2f'),
        'gap_pct': st.column_config.NumberColumn('Gap (%)', format='%+.0f'),
        'rate_2023': st.column_config.NumberColumn('2023 rate', format='%.2f'),
        'change_pct': st.column_config.NumberColumn('2023 vs 2020–2022 (%)', format='%+.0f'),
        'cases_2023': st.column_config.NumberColumn('Cases 2023', format='%d'),
        'outlook_2024': st.column_config.NumberColumn('2024 model outlook', format='%.2f'),
    })
    st.caption(f'Rates are {crime_name.lower()} cases per 1,000 people.')

    action_left, action_right, _ = st.columns([1.2, 1.4, 1.4])
    action_left.download_button('Download this list (CSV)', shown[columns].to_csv(index=False),
                                file_name=f'districts_to_review_{crime}.csv', mime='text/csv')
    chosen = action_right.selectbox('District', shown['district'], format_func=district_label, label_visibility='collapsed')
    action_right.button('Open district profile', on_click=go_to, args=('District profile', chosen))


# ---------- district profile ----------

def similar_districts(name, count=5):
    # districts with the closest profile: distance between the standardised model inputs (state left out)
    features = list(PROFILE_INPUTS)
    values = districts.set_index('district')[features]
    standardised = (values - values.mean()) / values.std()
    distance = np.sqrt(((standardised - standardised.loc[name]) ** 2).sum(axis=1))
    return distance.drop(name).sort_values().head(count).index


def next_step_text(row):
    if row['status'] == 'Review':
        return ('Crime here is above the expected level and rising. Put this district on the review shortlist: '
                'check local factors that the data does not cover, such as tourism, commuters, nightlife or specific '
                'hotspots, and whether current prevention work matches the trend.')
    if row['above_expected']:
        return ('Crime here is above the expected level but not rising sharply. Keep the district on the watch list and '
                'compare it with the similar districts below to see what differs.')
    if row['rising']:
        return ('Crime here is not above the expected level, but the 2023 rate rose. Check whether the rise '
                'continues when 2024 crime data is published.')
    return 'No flag at the current settings. The data does not suggest extra attention for this crime type.'


def profile_page():
    st.title('District profile')
    district_names = sorted(districts['district'])
    name = st.selectbox('District', district_names, index=district_names.index(st.session_state['selected_district']),
                        format_func=district_label, key='profile_district', on_change=remember_district, args=('profile_district',))
    row = table.set_index('district').loc[name]
    profile = districts.set_index('district').loc[name]
    other = 'property' if crime == 'assault' else 'assault'
    other_status = district_table(districts, years, other, above_pct, rising_pct, min_cases).set_index('district').loc[name, 'status']
    st.markdown(
        f'<p class="district-heading">{profile["state"]}. Profile group: {names[profile["cluster"]].lower()}.<br>'
        f'{crime_name} {status_badge(row["status"])} &nbsp; {CRIMES[other]} {status_badge(other_status)}</p>',
        unsafe_allow_html=True)

    test = year_ahead['test_results'].set_index(['target', 'model'])['R2']
    model_r2 = test.loc[f'{crime}_per_1000'].drop('baseline (2022 rate)').iloc[0]
    baseline_r2 = test.loc[(f'{crime}_per_1000', 'baseline (2022 rate)')]
    kind = STATUS_CLASSES[row['status']].replace('status', 'stat')
    show_stats([
        ('2020–2023 average', f'{row["average"]:.2f}', 'per 1,000 people', kind),
        ('Expected for this profile', f'{row["expected"]:.2f}', f'gap {signed(row["gap_pct"])}', kind),
        ('2023 rate', f'{row["rate_2023"]:.2f}', f'{signed(row["change_pct"])} vs 2020–2022, {row["cases_2023"]:,.0f} cases', kind),
        ('2024 model outlook', f'{row["outlook_2024"]:.2f}', 'year-ahead model', 'stat-quiet'),
    ])

    kind_class = kind if row['status'] != 'No flag' else ''
    st.markdown(f'<div class="next-step {kind_class}"><strong>Suggested next step</strong><p>{next_step_text(row)}</p></div>',
                unsafe_allow_html=True)
    if row['cases_2023'] < min_cases:
        st.caption(f'{name} recorded {row["cases_2023"]:.0f} cases in 2023, so its yearly rate can change a lot by chance.')

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader('Rate per year')
        district_rates = years[(years['district'] == name) & years['year'].between(2020, 2023)][['year', f'{crime}_per_1000']]
        district_rates = district_rates.rename(columns={f'{crime}_per_1000': 'rate'}).assign(series=name)
        state_rates = rates_by_year(years, 'state')
        state_rates = state_rates[(state_rates['state'] == profile['state']) & state_rates['year'].between(2020, 2023)]
        state_rates = state_rates[['year', f'{crime}_rate']].rename(columns={f'{crime}_rate': 'rate'}).assign(series=f'{profile["state"]} (state)')
        national = rates_by_year(years, None)
        national = national[national['year'].between(2020, 2023)][['year', f'{crime}_rate']].rename(columns={f'{crime}_rate': 'rate'}).assign(series='Malaysia')
        lines = pd.concat([district_rates, state_rates, national])
        order = [name, f'{profile["state"]} (state)', 'Malaysia']
        chart = alt.Chart(lines).mark_line(point=True).encode(
            x=alt.X('year:O', title=None), y=alt.Y('rate:Q', title='Per 1,000 people'),
            color=alt.Color('series:N', scale=alt.Scale(domain=order, range=LINE_COLOURS), title=None, sort=order),
            tooltip=['series', 'year', alt.Tooltip('rate:Q', format='.2f')])
        st.altair_chart(chart)
        st.caption(f'The 2024 outlook from the year-ahead model is {row["outlook_2024"]:.2f}. In the 2023 test, '
                   f'using the previous year’s rate was more accurate (R² {baseline_r2:.2f}) than this model '
                   f'(R² {model_r2:.2f}), so treat the outlook as a rough guide.')
    with right:
        st.subheader('Similar districts')
        similar = table.set_index('district').loc[similar_districts(name)].reset_index()
        st.dataframe(similar[['district', 'state', 'average', 'expected', 'status']], hide_index=True,
                     column_config={'district': 'District', 'state': 'State', 'status': 'Status',
                                    'average': st.column_config.NumberColumn('2020–2023 average', format='%.2f'),
                                    'expected': st.column_config.NumberColumn('Expected', format='%.2f')})
        st.caption('The five districts with the closest population, income, poverty, education and age profile.')

    st.subheader('Profile compared with all districts')
    rows = []
    for feature, (label, kind_of_value, _, _) in PROFILE_INPUTS.items():
        values = districts[feature]
        value, median = profile[feature], values.median()
        if kind_of_value == 'population':
            value, median = np.exp(value), np.exp(median)
        if kind_of_value == 'share':
            value, median = value * 100, median * 100
        rows.append({'Measure (2022)': label, name: value, 'Median district': median,
                     'Higher than (% of districts)': (values < profile[feature]).mean() * 100})
    st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        name: st.column_config.NumberColumn(format='%.2f'),
        'Median district': st.column_config.NumberColumn(format='%.2f'),
        'Higher than (% of districts)': st.column_config.ProgressColumn(format='%.0f', min_value=0, max_value=100)})

    report = (f'# {name}, {profile["state"]}: {crime_name.lower()}\n\n'
              f'Status: {row["status"]} (above expected by at least {above_pct}%, rising by at least {rising_pct}%)\n\n'
              f'- 2020–2023 average: {row["average"]:.2f} per 1,000 people\n'
              f'- Expected for this profile: {row["expected"]:.2f} (gap {signed(row["gap_pct"])})\n'
              f'- 2023 rate: {row["rate_2023"]:.2f} ({signed(row["change_pct"])} vs 2020–2022), {row["cases_2023"]:,.0f} cases\n'
              f'- 2024 year-ahead model outlook: {row["outlook_2024"]:.2f}\n'
              f'- Similar districts: {", ".join(similar["district"])}\n\n'
              f'Suggested next step: {next_step_text(row)}\n')
    st.download_button('Download district summary', report, file_name=f'{name}_{crime}_summary.md'.replace(' ', '_'))


# ---------- profile check ----------

def reset_profile(name):
    for feature in PROFILE_INPUTS:
        st.session_state.pop(f'check_{feature}_{name}', None)
    st.session_state.pop(f'check_state_{name}', None)


def profile_check_page():
    st.title('Check a district profile')
    st.markdown(
        '<p class="page-lead">Start from a district and change its profile, for example to reflect population '
        'growth or a new housing area, to see the crime rate that is typical for districts like it. The result '
        'shows a pattern in the data, not the effect of a policy.</p>', unsafe_allow_html=True)

    district_names = sorted(districts['district'])
    name = st.selectbox('Start from', district_names, index=district_names.index(st.session_state['selected_district']),
                        format_func=district_label, key='check_district', on_change=remember_district, args=('check_district',))
    profile = districts.set_index('district').loc[name]
    ranges = cross_district['ranges']

    # the results are shown above the inputs, so they stay in view while the profile is edited
    results_area = st.container()

    st.subheader('Profile')
    st.caption('Each value is limited to the range seen across the 131 districts.')
    edited = {}
    columns = st.columns(3)
    for i, (feature, (label, kind_of_value, step, number_format)) in enumerate(PROFILE_INPUTS.items()):
        low, high, value = ranges[feature]['min'], ranges[feature]['max'], profile[feature]
        if kind_of_value == 'population':
            low, high, value = np.exp(low), np.exp(high), np.exp(value)
        if kind_of_value == 'share':
            low, high, value = low * 100, high * 100, value * 100
        # the saved CSV and the model ranges can differ in the last decimal place, so the start value is kept inside the range
        value = min(max(value, low), high)
        entered = columns[i % 3].number_input(label, float(low), float(high), float(value), step=step, format=number_format,
                                              key=f'check_{feature}_{name}')
        if kind_of_value == 'population':
            entered = np.log(entered)
        if kind_of_value == 'share':
            entered = entered / 100
        edited[feature] = entered
    states = cross_district['states']
    edited['state'] = columns[0].selectbox('State', states, index=states.index(profile['state']), key=f'check_state_{name}')
    st.button(f'Reset to {name}', on_click=reset_profile, args=(name,))

    original_row = districts.set_index('district').loc[[name], cross_district['features']]
    edited_row = pd.DataFrame([edited])[cross_district['features']]
    with results_area:
        st.subheader('Typical rate for this profile')
        items = []
        for crime_type, label in CRIMES.items():
            model = cross_district['models'][f'{crime_type}_per_1000']
            original, new = model.predict(original_row)[0], model.predict(edited_row)[0]
            higher_than = (districts[f'{crime_type}_avg'] < new).mean() * 100
            items.append((f'{label} per 1,000 people', f'{new:.2f}',
                          f'{signed(new - original, 2, "")} vs {name}’s current profile; higher than {higher_than:.0f}% of districts’ actual averages'))
        show_stats(items)
        st.caption('This uses the cross-district model trained on all 131 districts. The expected rate on the '
                   'District profile page comes from a model trained without that district, so the two can differ slightly.')

        with st.expander('Year-ahead model for the same profile'):
            st.markdown('<p class="small-note">The year-ahead model uses four of these inputs (population, students, schools and '
                        'median income) and predicts a single year’s rate instead of a 2020–2023 average.</p>', unsafe_allow_html=True)
            row = pd.DataFrame([{feature: edited[feature] for feature in year_ahead['features']}])
            show_stats([(f'{label}, single year', f'{year_ahead["models"][f"{crime_type}_per_1000"].predict(row)[0]:.2f}',
                         year_ahead['model_names'][f'{crime_type}_per_1000'], 'stat-quiet') for crime_type, label in CRIMES.items()])


# ---------- about ----------

def about_page():
    st.title('About the data and models')
    st.markdown(
        '<p class="page-lead">Who this is for: crime-prevention planners, for example state police planning units or '
        'local safe-city programme teams, who decide which districts to look at first. The models give an expected '
        'rate and an outlook; the decision about what to do in a district stays with the planner.</p>',
        unsafe_allow_html=True)

    st.subheader('Cross-district model: the expected rate')
    repeated = cross_district['repeated_results']
    summary = repeated.groupby('target')['R2'].agg(['mean', 'min', 'max'])
    assault_model, property_model = cross_district['model_names']['assault_per_1000'], cross_district['model_names']['property_per_1000']
    chosen_models = (f'{assault_model} for both crime types' if assault_model == property_model
                     else f'{assault_model} for assault and {property_model} for property crime')
    st.markdown(
        f'- Predicts a district’s 2020–2023 average crime rate from its 2022 profile and state.\n'
        f'- Model: {chosen_models}, chosen with 5-fold cross-validation '
        f'from Linear Regression, Decision Tree, Random Forest, KNN and SVR.\n'
        f'- Tested on 10 random splits of the districts (80% training, 20% test). The mean test R² is '
        f'{summary.loc["assault_per_1000", "mean"]:.2f} for assault and {summary.loc["property_per_1000", "mean"]:.2f} for '
        f'property crime, so the profile explains part of the differences between districts, not all of them.')
    chart = alt.Chart(repeated.replace({'assault_per_1000': 'Assault', 'property_per_1000': 'Property crime'})).mark_line(point=True).encode(
        x=alt.X('split:O', title='Split'), y=alt.Y('R2:Q', title='Test R²'),
        color=alt.Color('target:N', scale=alt.Scale(range=LINE_COLOURS[:2]), title=None),
        tooltip=['target', 'split', 'model', alt.Tooltip('R2:Q', format='.2f')])
    st.altair_chart(chart)
    st.caption('Test R² for each of the 10 splits. The score depends on which districts end up in the 27-district test set.')

    st.subheader('Year-ahead model: the 2024 outlook')
    test = year_ahead['test_results'].copy()
    test['target'] = test['target'].map({'assault_per_1000': 'Assault', 'property_per_1000': 'Property crime'})
    st.markdown(
        '- Trained on 2022 and tested on 2023 for the same districts, then trained on 2022 and 2023 to predict 2024.\n'
        '- In the 2023 test, using each district’s previous-year rate was more accurate than the model, so the 2024 '
        'outlook is shown as a rough guide and is not used for the flags.')
    st.dataframe(test, hide_index=True, column_config={
        'target': 'Crime type', 'model': 'Model',
        'R2': st.column_config.NumberColumn('R²', format='%.2f'), 'MAE': st.column_config.NumberColumn(format='%.3f'),
        'MSE': st.column_config.NumberColumn(format='%.4f'), 'RMSE': st.column_config.NumberColumn(format='%.3f')})

    st.subheader('Limits of the data')
    st.markdown(
        '- Crimes are counted where they happen but divided by the people who live there, so districts with many '
        'commuters or tourists can show high rates.\n'
        '- There is no 2023 income survey; 2023 income is estimated from the 2022 and 2024 surveys.\n'
        '- Serdang’s 2020 and 2021 property crime totals were corrected in the source data (other theft was counted twice).\n'
        '- Crime data ends in 2023, so the 2024 outlook cannot be checked yet.\n'
        '- The models show patterns between districts, not causes.')


{'Overview': overview_page, 'Districts to review': review_page, 'District profile': profile_page,
 'Profile check': profile_check_page, 'About the data and models': about_page}[page]()
