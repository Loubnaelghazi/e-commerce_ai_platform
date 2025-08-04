
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import random
from typing import Dict, Any, List
import sys
from src.mcp.server import MCPServer


class DataCollectorAgent:

    def __init__(self):

        self.name = "DataCollector"
        self.server = MCPServer(self.name, port=8001)
        self.data_cache = {}
        
        self.server.add_tool("collect_customer_data", self.collect_customer_data)
        self.server.add_tool("collect_transaction_data", self.collect_transaction_data)
        self.server.add_tool("clean_data", self.clean_data)
        self.server.add_tool("detect_anomalies", self.detect_anomalies)
    


    async def collect_customer_data(self, args: Dict[str, Any]) -> Dict[str, Any]:

        """la collecte de donnees clients depuis diverses sources ici generation aleatoire """

        print(f"[{self.name}] Collecting customer data...")
        
        n_customers = args.get('n_customers', 1000)
        
        customers = []
        for i in range(n_customers):

            customer = {
                'customer_id': f'CUST_{i:06d}',
                'age': np.random.randint(18, 80),
                'gender': np.random.choice(['M', 'F']),

                'city': np.random.choice(['Rabat', 'Nador', 'Tangier', 'Meknes']),
                'registration_date': (datetime.now() - timedelta(days=np.random.randint(1, 365*3))).isoformat(),
                'email_validated': np.random.choice([True, False], p=[0.8, 0.2]),
                'newsletter_subscribed': np.random.choice([True, False], p=[0.3, 0.7])}
          
          
            customers.append(customer)
        
        self.data_cache['customers'] = pd.DataFrame(customers)
        
        return {
            'status': 'success',
            'message': f'Collected {n_customers} customer records',
            'data_key': 'customers',
            'records_count': n_customers
        }
    
    async def collect_transaction_data(self, args: Dict[str, Any]) -> Dict[str, Any]:

        """Simule la collecte de donnees de transactions"""

        print(f"[{self.name}] Collecting transaction data...")
        
        if 'customers' not in self.data_cache:
            return {'status': 'error', 'message': 'Customer data not available. Collect customers first.'}
        
        customers_df = self.data_cache['customers']

        n_transactions = args.get('n_transactions', 5000)
        
        transactions = []

        for i in range(n_transactions):
            customer_id = np.random.choice(customers_df['customer_id'])
            
            transaction = {
                'transaction_id': f'TXN_{i:08d}',
                'customer_id': customer_id,
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports']),
                'amount': np.round(np.random.lognormal(4, 1), 2),
                'quantity': np.random.randint(1, 10),
                'transaction_date': (datetime.now() - timedelta(days=np.random.randint(1, 90))).isoformat(),
                'payment_method': np.random.choice(['Card', 'PayPal', 'Bank Transfer']),
                'discount_applied': np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05])
            }
            transactions.append(transaction)
        
        self.data_cache['transactions'] = pd.DataFrame(transactions)
        
        return {
            'status': 'success',
            'message': f'Collected {n_transactions} transaction records',
            'data_key': 'transactions',
            'records_count': n_transactions
        }
    
    async def clean_data(self, args: Dict[str, Any]) -> Dict[str, Any]:

        """Nettoie les donnees collectees"""
        print(f"[{self.name}] Cleaning data...")
        
        data_key = args.get('data_key')
        if data_key not in self.data_cache:
            return {'status': 'error', 'message': f'Data key {data_key} not found'}
        
        df = self.data_cache[data_key].copy()
        original_rows = len(df)
        
        
        if data_key == 'customers':

            df = df.drop_duplicates(subset=['customer_id'])
            
            df = df[(df['age'] >= 18) & (df['age'] <= 100)]
            df['city'] = df['city'].str.title()
            
        elif data_key == 'transactions':
            df = df.drop_duplicates(subset=['transaction_id'])
            
            df = df[df['amount'] > 0]
            
            df = df[df['quantity'] > 0]
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        cleaned_key = f"{data_key}_cleaned"
        self.data_cache[cleaned_key] = df
        
        cleaned_rows = len(df)
        removed_rows = original_rows - cleaned_rows
        
        return {
            'status': 'success',
            'message': f'Data cleaned. Removed {removed_rows} invalid records',
            'data_key': cleaned_key,
            'original_rows': original_rows,
            'cleaned_rows': cleaned_rows,
            'removed_rows': removed_rows
        }
    



    async def detect_anomalies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[{self.name}] Detecting anomalies...")
        
        data_key = args.get('data_key')
        if data_key not in self.data_cache:
            return {'status': 'error', 'message': f'Data key {data_key} not found'}
        
        df = self.data_cache[data_key]
        anomalies = []
        
        if 'transactions' in data_key:

            # (montants trees eleves)

            amount_threshold = df['amount'].quantile(0.99)
            high_amount_anomalies = df[df['amount'] > amount_threshold]
            
            quantity_threshold = df['quantity'].quantile(0.95)
            high_quantity_anomalies = df[df['quantity'] > quantity_threshold]
            
            anomalies = {
                'high_amount_transactions': len(high_amount_anomalies),
                'high_quantity_transactions': len(high_quantity_anomalies),
                'amount_threshold': amount_threshold,
                'quantity_threshold': quantity_threshold
            }
        
        return {
            'status': 'success',
            'message': 'Anomaly detection completed',
            'anomalies': anomalies,
            'total_records': len(df)
        }
    

    def get_data(self, key: str) -> pd.DataFrame:

        """ helper """
        return self.data_cache.get(key)
    



    async def start(self):
        """Démarre l'agent"""
        print(f"[{self.name}] Starting agent...")
        await self.server.start()

async def test_data_collector():
    agent = DataCollectorAgent()
    
    server_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)  
    
    print("\n=== Test Data Collector Agent ===")
    
    # Test collecte clients
    result = await agent.collect_customer_data({'n_customers': 100})
    print(f"Customer collection: {result}")
    
    # Test collecte transactions
    result = await agent.collect_transaction_data({'n_transactions': 500})
    print(f"Transaction collection: {result}")
    
    # Test nettoyage
    result = await agent.clean_data({'data_key': 'customers'})
    print(f"Customer cleaning: {result}")
    
    result = await agent.clean_data({'data_key': 'transactions'})
    print(f"Transaction cleaning: {result}")
    
    # Test detection d'anomalies
    result = await agent.detect_anomalies({'data_key': 'transactions_cleaned'})
    print(f"Anomaly detection: {result}")
    
    # echantillon des donnees
    customers = agent.get_data('customers_cleaned')
    transactions = agent.get_data('transactions_cleaned')
    
    print(f"\nCustomers sample:\n{customers.head()}")
    print(f"\nTransactions sample:\n{transactions.head()}")


if __name__ == "__main__":
    asyncio.run(test_data_collector())