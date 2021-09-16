import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Return a merged dataframe of messages and categories"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id') # merge datasets
    return df

def clean_data(df):
    """Return a cleaned dataframe with category columns"""
    categories = df['categories'].str.split(pat=';', expand=True)
    rows = categories.loc[1,:] # select the first row of the categories dataframe
    category_colnames = [n[0:-2] for n in rows] # extract category names
    categories.columns = category_colnames # rename the columns of `categories`
    
    for column in categories:
        categories[column] = categories[column].str[-1] # set each value to be the last character of the string
        categories[column] = categories[column].astype(int) # convert column from string to numeric
    
    df = df.drop('categories', axis=1) # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe
    
    df = df.drop_duplicates() # drop duplicates
    
    return df

def save_data(df, database_filename):
    """Save the clean dataset into an SQLite database."""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disasterTable', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()