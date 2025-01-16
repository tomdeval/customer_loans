import yaml
from sqlalchemy import create_engine
import pandas as pd

def load_credentials(file_path='credentials.yaml'):
    with open(file_path, 'r') as cred:
        credentials = yaml.safe_load(cred)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.credentials = credentials
        self.engine = None
    
    def initialize_engine(self):
        db_url = (
            f"postgresql://{self.credentials['RDS_USER']}:"
            f"{self.credentials['RDS_PASSWORD']}@"
            f"{self.credentials['RDS_HOST']}:"
            f"{self.credentials['RDS_PORT']}/"
            f"{self.credentials['RDS_DATABASE']}"
        )
        self.engine = create_engine(db_url)
        print("Engine initialized successfully.")
        return self.engine
    
    def extract_data_to_dataframe(self, table_name='loan_payments'):
        if self.engine is None:
            raise ValueError("Engine is not initialized. Call initialize_engine() first.")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.engine)
        print(f"Data extracted from table '{table_name}' successfully.")
        return df
    
def save_dataframe_to_csv(dataframe, file_name='loan_payments.csv'):
    dataframe.to_csv(file_name, index=False)
    print(f"Data saved to file '{file_name}' successfully.")

if __name__ == '__main__':
    credentials_path = 'credentials.yaml'
    table_name = 'loan_payments'
    output_file = 'loan_payments.csv'

    try:
        # Load credentials
        creds = load_credentials(credentials_path)

        # Initialize database connector
        connector = RDSDatabaseConnector(credentials=creds)
        connector.initialize_engine()

        # Extract data to dataframe
        df = connector.extract_data_to_dataframe(table_name)

        # Save dataframe to CSV
        save_dataframe_to_csv(df, output_file)
    except Exception as e:
        print(f"An error occurred: {e}")