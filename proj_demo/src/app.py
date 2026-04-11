import streamlit as st
import requests

import time
import plotly.graph_objects as go

st.set_page_config(
    page_title='Iris Classfication System',
    page_icon='🌸',
    layout='wide',
    initial_sidebar_state='expanded',
)

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:8001"

def call_api(endpoint, method='GET', payload=None):
    url = f"{st.session_state.api_url}/{endpoint}"

    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=10)
        else:
            response = requests.get(url, timeout=5)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f'API Error: {response.status_code} - {response.text}'

    except requests.exceptions.ConnectionError:
        return None, 'Connection Error'

    except Exception as e:
        return None, f'API Error: {str(e)}'

def render_sidebar():
    with st.sidebar:
        st.header('Controls')
        data, error = call_api('health')

        if data and data.get('status') == "healthy":
            st.success('✅ API is healthy')
        else:
            st.error("❌ API is unhealthy")

        uptime = time.time() - st.session_state.start_time
        st.markdown("### Uptime")
        st.markdown(f"# {uptime}s")

        st.divider()

        selected_page = st.sidebar.selectbox(
            'select page',
            ['Single Page', 'Batch Analysis', "Model Monitoring"],
        )
        return selected_page

def page_single_prediction():

    st.header('Single Prediction')
    col_input, col_pred = st.columns([1.5, 1])

    with col_input:
        st.subheader("Input Parameters")
        sepal_length_str = st.text_input("Sepal Length (cm)", value = "5.8")

        try:
            sepal_length = float(sepal_length_str)
        except ValueError:
            st.error("Please enter a numeric value")
            sepal_length = 5.8
        sepal_width_options = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        sepal_width = st.selectbox('Sepal Width()', options=sepal_width_options, index=2)

        petal_length = st.slider('Petal length (cm)', 1.0, 7.0, 3.8)
        petal_width = st.slider('Petal width (cm)', 0.1, 2.5, 1.2)

        with st.expander('> Advanced Options'):
            st.selectbox('Model Version', ["v1.0.0(Prediction)", 'v1.1.0(staging)'])
    
    with col_pred:
        st.subheader("Prediction")
        predict_btn = st.button('Predict', type = "primary", use_container_width=True)

        if predict_btn:
            payload = {
                'sepal_length' : sepal_length,
            }

            with st.spinner('Analyzing...'):
                start_ts = time.time()
                result, error = call_api('predict', method="POST", payload=)

                end_ts = time.time()
            if result:
                latency = (end_ts - start_ts) * 1000
                pred_class = result.get('prediction')
                prob = result.get('probability', 0.0)
                ocnf = result.get('confidence', "Low")

                st.success(f"Prediction Complete !")

                fig = go.Figure (
                    go.Indicator(
                        mode = "gauge+number",
                        value=prob,
                        domain={
                            "x": [0, 1],
                            "y": [0, 1],
                        },
                        title={'text': f'Probability({pred_class})'},
                        gauge={
                            'axis':{
                                'range': [0,1], 
                                'tickwidth': 1
                            },
                            'bar': {'color': 'green'},
                            'steps': [
                                {'range': [0, 0.5], 'color':'lightgray'},
                                {'range': [0.5, 0.8], 'color':'lightgreen'},
                                {'range': [0.8, 1], 'color':'skyblue'},
                            ],
                            'threshold':{
                                'line':{'color':'red','width':4},
                                'thickness': 0.75,
                                'value': prob
                            }
                        }
                    )
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                m1, m2, m3 = st.columns(3)
                m1.metric('Probability', f'{prob:.1%}')
                m2.metric('confidence', conf)
                m3.metric('Latency', f'{latency:.0f}ms')
            
            else:
                st.error(error)

def page_batch_analy():
    st.header('Batch Analysis')

def page_model_monitoring():
    st.header('model_monitoring')

def main():
    st.title("Iris Classification System")

    current_page = render_sidebar()
    st.divider()
    if current_page == "Single Prediction":
        page_single_prediction()
    elif current_page == "Model Monitoring":
        page_model_monitoring()

if __name__ == "__main__":
    main()