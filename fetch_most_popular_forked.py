import pandas as pd
import requests
import os
from dotenv import load_dotenv


# Load the CSV file
users_df = pd.read_csv('users_final_with_forked_status.csv')


# Remove rows that don't have a 'most_popular_repo_forked' column
if 'most_popular_repo_forked' in users_df.columns:
    users_df['most_popular_repo_forked'].fillna(False, inplace=True)
else:
    users_df = pd.DataFrame()  # Empty DataFrame if the column doesn't exist


#write the DataFrame to a CSV file
users_df.to_csv('data/users_removed.csv', index=False)