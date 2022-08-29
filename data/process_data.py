# import libraries

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


# Creates a function to load the data from the csv files
def load_data(messages_path, categories_path):
    """Load data from paths
    INPUT
    messages_path -- path to messages file
    categories_path -- path to categories file
    OUTPUT
    df - pandas DataFrame
    """
    
    print('Loading data...')
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = messages.merge(categories, how = 'left', on = 'id')
    
    return df


# Creates a function that cleans the data 
def clean_data(df):
    """ Clean data 
    INPUT
    df -- type pandas DataFrame
    OUTPUT
    df -- cleaned pandas DataFrame
    """    
    
    print('Cleaning data...')
    # Split `categories` into separate category columns.
    categories = df.categories.str.split(pat = ';', expand = True)
    # Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[1,:]
    
    # Rename columns of `categories` with new column names.
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column]
    
    
    # drop the original categories column from `df` and concat the dataframes
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1, sort=False)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    print('Duplicates left:', df.duplicated().sum())
    
    return df


# Creates a function that saves the dataframe in a database
def save_data(df, database_filename):
    """Saves DataFrame to database
    INPUT
    df -- type pandas DataFrame
    database_filename -- name of the database to be used 
    """
    name = 'sqlite:///' + database_filename
    # create database
    engine = create_engine(name)
    df.to_sql('clean_dataset', engine, index=False)
    print('Saving database...')

    
# The main function        
def main():
    """Runs main functions"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        df = load_data(messages_filepath, categories_filepath)
        df = clean_data(df)
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
    
    else:
        print('Error, missing file paths')


if __name__ == '__main__':
    main()




