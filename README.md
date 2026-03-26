This is the complete workflow that allowed me to create an operating software able to find value bets in the top 5 european football leagues.

The pipeline was conducted as follows:
- obtaining data from different football APIs in order to get historical data on the odds of several matches and key features of the teams involved in such matches
- modelling phase
- obtaining the same data of before but just for the upcoming matches
- developing of the final software


The final software presents all the upcoming matches with the following information:
- most probable outcome
- its probability
- the odds
- the Expected Value (product of the odd and its probability)

The matches are ordered in decreasing order of Expected Value and the ones with profitable Expected Value (greater than 1.05) are highlighted. 
There is also a second page where the user can select a number of profitable matches and the sofware returns the best combination of matches with the highest Expected Value or the highest probability out of all the possible combinations. 

To start the software just set as current directory the directory in which the files had been downloaded and then execute the command streamlit run main.py.
