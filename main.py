import pandas as pd
import streamlit as st
import numpy as np
from predictions import *


df = final_data_frame.sort_values(by='Expected Value', ascending=False)


st.set_page_config(page_title="Value Betting Dashboard", layout="wide", initial_sidebar_state="expanded")


page = st.sidebar.radio("Select what to do:",
    ["1. Value Bets Dashboard", "2. Optimization Multiple Bets"])


# Let's create the first function of the final software: higlighting value bets
if page == "1. Value Bets Dashboard":
    st.title("Odds exploration and Value Bets")
    
    def highlight_value_bets(row):
        if row['Expected Value'] >= 1.05:
            return ['background-color: #98FB98'] * len(row) 
        return [''] * len(row)
    
    st.dataframe(df.style.apply(highlight_value_bets, axis=1), use_container_width=True)


# Let's create the second function of the software: select the n matches with highest combined expected value and combined probability
elif page == "2. Optimization Multiple Bets":
    st.title("Multiple Bets Generator (Accumulator)")

    value_bets = df[df['Expected Value'] >= 1.05].copy()

    if value_bets.empty:
        print('No value bets for this round')

    else:
        num_games = st.slider('How many games do you want to insert? ', max_value=10)

        highest_exp_value = value_bets.sort_values(by='Expected Value', ascending=False).head(num_games)
        tot_odd_1 = highest_exp_value['Bet'].prod()
        tot_prob_1 = highest_exp_value['Probability'].prod()
        total_ev_1 = tot_odd_1 * tot_prob_1

        st.subheader("Summary of multiple bet with highest EV:")
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Odds", f"{tot_odd_1:.2f}")
        col2.metric("Win Probability", f"{tot_prob_1:.1%}")
        col3.metric("Total Expected Value", f"{total_ev_1:.2f}")
        
        st.markdown("### Selected Matches")
        bet_columns = ['Game', 'Forecasted result', 'Bet', 'Probability', 'Expected Value']
        
        st.dataframe(
            highest_exp_value[bet_columns].style.format({
                'Odd': '{:.2f}',
                'Probability': '{:.1%}',
                'Expected Value': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )

        

        highest_probability = df.sort_values(by='Probability', ascending=False).head(num_games)
        tot_odd_2 = highest_probability['Bet'].prod()
        tot_prob_2 = highest_probability['Probability'].prod()
        total_ev_2 = tot_odd_2 * tot_prob_2

        st.subheader("Summary of multiple bet with highest win probability")
            
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Total Odd", f"{tot_odd_2:.2f}")
        col5.metric("Win Probability", f"{tot_prob_2:.1%}")
        col6.metric("Total Expected Value", f"{total_ev_2:.2f}")
        
        st.markdown("### Selected Matches")
       
        colonne_bolletta = ['Game', 'Forecasted result', 'Bet', 'Probability', 'Expected Value']
        
        st.dataframe(
            highest_probability[colonne_bolletta].style.format({
                'Odd': '{:.2f}',
                'Probability': '{:.1%}',
                'Expected Value': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )


