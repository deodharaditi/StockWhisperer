import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from main import predict

def add_dropdown():
    options = ["PPL Corporation (PPL)", "Kimberly-Clark Corporation (KMB)", "Oil-Dri Corporation of America (ODC)",
            "Eversource Energy (ES)", "Becton, Dickinson and Company (BDX)", "Penns Woods Bancorp, Inc. (PWOD)",
            "Healthcare Realty Trust Incorporated (HR)", "Kellanova (K)", "Southwest Gas Holdings, Inc. (SWX)",
            "Sirius XM Holdings Inc. (SIRI)"]

    try:
        selected_company = st.selectbox("Which stock would you like the predictions for?", options, index=None, placeholder="PPL Corporation (PPL)")
        selected_company = selected_company.split("(")[1].split(")")[0]
    except:
        selected_company = "PPL"
    
    return selected_company


def add_sidebar(company):

    data = get_data(company)
    # print(data)
    st.sidebar.header("Model Paramters")

    slider_labels = [
            ("Revenue (MM)", "revenue"),
            ("Earning per Share Basic ($)", "eps_basic"),
            ("Cash and Equivalent Assets (MM)", "cash_and_equiv"),
            ("Total Assets (MM)", "total_assets"),
            ("Total Liabilities (MM)", "total_liabilities"),
            ("Retained Earnings (MM)", "retained_earnings"),
            ("Total Liabilities and Equity (MM)", "total_liabilities_and_equity"),
            ("Operation Cash Flow (MM)", "cf_cfo"),
            ("Total Assets Growth", "total_assets_growth"),
            ("Equity to Assets Ratio", "equity_to_assets"),
        ]
    
    input_dict = {}


    for label, key in slider_labels:
        if data[key].max() > 1000000:
            input_dict[key] = st.sidebar.slider(label,
                                                min_value=float(0),
                                                max_value=float(data[key].max())/1000000,
                                                value=float(data[key][74])/1000000,
                                                key=f"{key}_{company}"
        )
        else:
            input_dict[key] = st.sidebar.slider(label,
                                                min_value=float(0),
                                                max_value=float(data[key].max()),
                                                value=float(data[key][74]),
                                                key=f"{key}_{company}"
        )

    return input_dict


def get_data(company):
  
  data = pd.read_csv("reference.csv")

  filtered_data = data[data['Symbol'] == company]
  filtered_data = filtered_data.reset_index(drop=True, inplace=False)

  return filtered_data


def add_predictions(input_data, selected_company):
    #print(input_data)
    prediction = predict(input_data, selected_company)
    
    st.subheader("Stock prediction")
    st.write("The predicted value of the stock ($) for the next quarter is:", prediction)


def plot_parameter_comparison(input_data, selected_company):
    # Create a DataFrame from the input_data dictionary
    df = pd.DataFrame(input_data, index=[0])

    # Transpose the DataFrame so that each parameter is a column
    df = df.T.reset_index()
    df.columns = ['Parameter', 'Value']

    # Plot the parameter comparison using Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(x=df['Parameter'], y=df['Value'], name=selected_company))

    fig.update_layout(title=f"Parameter Comparison for {selected_company}",
                      xaxis_title="Parameter",
                      yaxis_title="Value")

    return fig

def plot_parameter_radar(input_data, selected_company):
    # Create a DataFrame from input data
    df = pd.DataFrame(input_data, index=[0])

    # Plot parameter radar chart
    categories = list(df.columns)
    values = df.values[0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=selected_company
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df.max().max()]
            )),
        showlegend=True,
        title=f"Parameter Radar Chart for {selected_company}"
    )
    
    return fig


def plot_parameter_trends(input_data, selected_company, start_year, end_year):
    # Filter data based on selected year range
    filtered_data = input_data[(input_data['period_end_date'].str[:4].astype(int).between(start_year, end_year))]
    filtered_data = filtered_data[filtered_data['Symbol'] == selected_company]
    print(filtered_data)
    # Plot parameter trends over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['period_end_date'], y=filtered_data['Average_Stock_Value_After_Result'], mode='lines', name='Average Stock Value'))

    fig.update_layout(title=f"Average Stock Value Over Time for {selected_company}",
                      xaxis_title="Time",
                      yaxis_title="Average Stock Value (USD)")
    
    return fig

def main():
  
  df = pd.read_csv("reference.csv")
  start_year = 0
  end_year = 0

  st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  st.title('Stock Whisperer')
  st.write("Simply select your preferred stock from a curated list, explore key financial metrics, and receive accurate forecasts to guide your investment strategy. Whether you're a seasoned investor or new to the stock market, our app provides valuable insights to optimize your portfolio and maximize returns. Stay ahead of the curve and take control of your investments with the Stock Market Predictor app today!")
  
  selected_company  = add_dropdown()

  input_data = add_sidebar(selected_company)
  in_data = ['revenue', 'cash_and_equiv', 'total_assets', 'total_liabilities', 'retained_earnings', 'total_liabilities_and_equity', 'cf_cfo']
  for k,v in input_data.items():
      if k in in_data:
          input_data[k] *= 1000000
  #print(input_data)

  with st.container():
    
    col1, col2 = st.columns([4,1])

    with col2:
        add_predictions(input_data, selected_company)

        start_year = st.slider("Select Start Year", min_value=2002, max_value=2019, value=2002)
        end_year = st.slider("Select End Year", min_value=start_year, max_value=2019, value=2019)

    with col1:
       st.plotly_chart(plot_parameter_comparison(input_data, selected_company))

       st.plotly_chart(plot_parameter_radar(input_data, selected_company))

       fig = plot_parameter_trends(df, selected_company, start_year, end_year)
       st.plotly_chart(fig)

if __name__ == '__main__':
  main()
