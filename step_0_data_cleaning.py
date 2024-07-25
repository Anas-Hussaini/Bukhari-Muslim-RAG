# import pandas library 
import pandas as pd

# Load the CSV file into a Pandas DataFrame
all_hadith_df = pd.read_csv('all_hadiths_clean.csv')

# Preview the data
print(all_hadith_df.head())

# Check colmuns names to see which to drop and keep
all_hadith_df.columns

# Drop irrelevant columns
all_hadith_df = all_hadith_df.drop(columns=['id', 'hadith_id', 'chapter_no', 'chapter',
       'chain_indx', 'text_ar'])

# Print to check dropped columns
print(all_hadith_df.head)

# Drop rows with missing values
all_hadith_df.dropna()

# Check unique values in source column to see which to drop and keep
all_hadith_df['source'].unique()

# Drop irrelevant source books
bukhari__muslim_df = all_hadith_df[all_hadith_df["source"].str.contains("Sunan Abi Da'ud|Jami' al-Tirmidhi|Sunan an-Nasa'i|Sunan Ibn Majah") == False]

# Check dropped source books
bukhari__muslim_df['source'].unique()

# Print to check dropped rows of irrelevant source books
bukhari__muslim_df

# Save cleaned data to a new csv file
bukhari__muslim_df.to_csv('bukhari_muslim.csv', index=False)

##### Only to take a small sample of dataframe
# first_50 = bukhari__muslim_df [:50]
# first_50.to_csv('first_50.csv', index=False)