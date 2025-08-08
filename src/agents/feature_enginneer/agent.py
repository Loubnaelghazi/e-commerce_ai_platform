import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.mcp.server import MCPServer
from src.mcp.client import MCPClient

class FeatureEngineerAgent:

    def __init__(self):

        self.name = "FeatureEngineer"
        self.server = MCPServer(self.name, port=8002)
        self.client = MCPClient(self.name)
        self.feature_cache = {}
        self.encoders = {}
        self.scalers = {}
        #
        self.server.add_tool("create_customer_features", self.create_customer_features)
        self.server.add_tool("create_rfm_features", self.create_rfm_features)
        self.server.add_tool("create_behavioral_features", self.create_behavioral_features)
        self.server.add_tool("prepare_ml_dataset", self.prepare_ml_dataset)
        
    async def create_customer_features(self, args: Dict[str, Any]) -> Dict[str, Any]:

        """Crée des features basiques des clients via MCP avec DataCollector"""
        print(f"[{self.name}] Creating customer features...")
        
        try:
            # Appel MCP au DataCollector pour obtenir les données clients nettoyées
            customers_result = await self.client.call_tool(
                "DataCollector", 
                "clean_data", 
                {"data_key": "customers"}
            )
            

            
            if customers_result.get('status') != 'success':
                return {'status': 'error', 'message': 'Failed to get clean customer data'}
            
            
            # Simulation de récupération des données (dans un vrai système, on récupérerait via MCP)
            # Pour cette démo, on crée des données similaires
            n_customers = 1000
            customers_data = []
            for i in range(n_customers):
                customer = {
                    'customer_id': f'CUST_{i:06d}',
                    'age': np.random.randint(18, 80),
                    'gender': np.random.choice(['M', 'F']),
                    'city': np.random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice']),
                    'registration_date': (datetime.now() - timedelta(days=np.random.randint(1, 365*3))).isoformat(),
                }
                customers_data.append(customer)
            
            customers_df = pd.DataFrame(customers_data)
            customers_df['registration_date'] = pd.to_datetime(customers_df['registration_date'])
            
            # Création des features
            features_df = customers_df.copy()
            
            # Feature: Ancienneté en jours
            features_df['days_since_registration'] = (datetime.now() - features_df['registration_date']).dt.days
            
            # Feature: Tranche d'âge
            features_df['age_group'] = pd.cut(
                features_df['age'], 
                bins=[0, 25, 35, 50, 65, 100], 
                labels=['18-25', '26-35', '36-50', '51-65', '65+']
            )
            
            # Encodage des variables catégorielles
            le_gender = LabelEncoder()
            features_df['gender_encoded'] = le_gender.fit_transform(features_df['gender'])
            self.encoders['gender'] = le_gender
            
            le_city = LabelEncoder()
            features_df['city_encoded'] = le_city.fit_transform(features_df['city'])
            self.encoders['city'] = le_city
            
            # Feature: Score d'ancienneté (0-1)
            max_days = features_df['days_since_registration'].max()
            features_df['loyalty_score'] = features_df['days_since_registration'] / max_days
            
            # Sauvegarde
            self.feature_cache['customer_features'] = features_df
            
            return {
                'status': 'success',
                'message': f'Created customer features for {len(features_df)} customers',
                'features_created': ['days_since_registration', 'age_group', 'gender_encoded', 'city_encoded', 'loyalty_score'],
                'data_key': 'customer_features'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error creating customer features: {str(e)}'}
    
    async def create_rfm_features(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Crée les features RFM (Récence, Fréquence, Montant)"""
        print(f"[{self.name}] Creating RFM features...")
        
        try:
            # Simulation des données de transaction pour la démo
            n_transactions = 5000
            customer_ids = [f'CUST_{i:06d}' for i in range(1000)]
            
            transactions_data = []
            for i in range(n_transactions):
                transaction = {
                    'transaction_id': f'TXN_{i:08d}',
                    'customer_id': np.random.choice(customer_ids),
                    'amount': np.round(np.random.lognormal(4, 1), 2),
                    'transaction_date': datetime.now() - timedelta(days=np.random.randint(1, 365))
                }
                transactions_data.append(transaction)
            
            transactions_df = pd.DataFrame(transactions_data)
            
            # Calcul des features RFM par client
            current_date = datetime.now()
            
            rfm_features = transactions_df.groupby('customer_id').agg({
                'transaction_date': [
                    lambda x: (current_date - x.max()).days,  # Recency
                    'count'  # Frequency
                ],
                'amount': [
                    'sum',     # Monetary total
                    'mean',    # Monetary moyenne
                    'std'      # Écart-type des montants
                ]
            }).round(2)
            
            # Flatten column names
            rfm_features.columns = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'monetary_std']
            rfm_features = rfm_features.fillna(0)
            
            # Scores RFM (1-5)
            rfm_features['recency_score'] = pd.qcut(
                rfm_features['recency_days'], 
                q=5, 
                labels=[5,4,3,2,1],  # Plus récent = score plus élevé
                duplicates='drop'
            ).astype(float)
            
            rfm_features['frequency_score'] = pd.qcut(
                rfm_features['frequency'], 
                q=5, 
                labels=[1,2,3,4,5],  # Plus fréquent = score plus élevé
                duplicates='drop'
            ).astype(float)
            
            rfm_features['monetary_score'] = pd.qcut(
                rfm_features['monetary_total'], 
                q=5, 
                labels=[1,2,3,4,5],  # Plus de dépenses = score plus élevé
                duplicates='drop'
            ).astype(float)
            
            # Score RFM combiné
            rfm_features['rfm_score'] = (
                rfm_features['recency_score'] + 
                rfm_features['frequency_score'] + 
                rfm_features['monetary_score']
            ) / 3
            
            # Segmentation client basée sur RFM
            def rfm_segment(row):
                if row['rfm_score'] >= 4.5:
                    return 'Champions'
                elif row['rfm_score'] >= 3.5:
                    return 'Loyal Customers'
                elif row['rfm_score'] >= 2.5:
                    return 'Potential Loyalists'
                elif row['rfm_score'] >= 1.5:
                    return 'At Risk'
                else:
                    return 'Lost Customers'
            
            rfm_features['customer_segment'] = rfm_features.apply(rfm_segment, axis=1)
            
            # Reset index pour avoir customer_id comme colonne
            rfm_features = rfm_features.reset_index()
            
            self.feature_cache['rfm_features'] = rfm_features
            
            return {
                'status': 'success',
                'message': f'Created RFM features for {len(rfm_features)} customers',
                'features_created': ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'rfm_score', 'customer_segment'],
                'segments_distribution': rfm_features['customer_segment'].value_counts().to_dict(),
                'data_key': 'rfm_features'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error creating RFM features: {str(e)}'}
    
    async def create_behavioral_features(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Crée des features comportementales avancées"""
        print(f"[{self.name}] Creating behavioral features...")
        
        try:
            # Utilise les transactions déjà générées ou en crée de nouvelles
            if 'rfm_features' not in self.feature_cache:
                await self.create_rfm_features({})
            
            # Simulation de données comportementales
            customer_ids = [f'CUST_{i:06d}' for i in range(1000)]
            
            behavioral_data = []
            for customer_id in customer_ids:
                behavior = {
                    'customer_id': customer_id,
                    'avg_session_duration': np.random.exponential(15),  # minutes
                    'pages_per_session': np.random.poisson(5),
                    'bounce_rate': np.random.beta(2, 5),  # Entre 0 et 1
                    'cart_abandonment_rate': np.random.beta(3, 7),
                    'email_open_rate': np.random.beta(4, 6),
                    'social_media_engagement': np.random.poisson(2),
                    'support_tickets': np.random.poisson(1),
                    'product_categories_visited': np.random.randint(1, 8)
                }
                behavioral_data.append(behavior)
            
            behavioral_df = pd.DataFrame(behavioral_data)
            
            # Features dérivées
            behavioral_df['engagement_score'] = (
                behavioral_df['avg_session_duration'] * 0.3 +
                behavioral_df['pages_per_session'] * 0.2 +
                (1 - behavioral_df['bounce_rate']) * 0.2 +
                behavioral_df['email_open_rate'] * 0.2 +
                behavioral_df['social_media_engagement'] * 0.1
            )
            
            # Score de satisfaction (inverse des tickets support + engagement)
            behavioral_df['satisfaction_score'] = np.maximum(
                0, 
                behavioral_df['engagement_score'] - behavioral_df['support_tickets'] * 0.5
            )
            
            # Diversité des intérêts
            behavioral_df['interest_diversity'] = behavioral_df['product_categories_visited'] / 8
            
            # Score de risque de churn
            behavioral_df['churn_risk_score'] = (
                behavioral_df['cart_abandonment_rate'] * 0.4 +
                behavioral_df['bounce_rate'] * 0.3 +
                (1 - behavioral_df['email_open_rate']) * 0.2 +
                (behavioral_df['support_tickets'] / 5) * 0.1
            )
            
            self.feature_cache['behavioral_features'] = behavioral_df
            
            return {
                'status': 'success',
                'message': f'Created behavioral features for {len(behavioral_df)} customers',
                'features_created': ['engagement_score', 'satisfaction_score', 'interest_diversity', 'churn_risk_score'],
                'avg_engagement': float(behavioral_df['engagement_score'].mean()),
                'avg_churn_risk': float(behavioral_df['churn_risk_score'].mean()),
                'data_key': 'behavioral_features'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error creating behavioral features: {str(e)}'}
    
    async def prepare_ml_dataset(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Combine toutes les features pour créer le dataset ML final"""
        print(f"[{self.name}] Preparing ML dataset...")
        
        try:
            # S'assurer que toutes les features sont créées
            if 'customer_features' not in self.feature_cache:
                await self.create_customer_features({})
            if 'rfm_features' not in self.feature_cache:
                await self.create_rfm_features({})
            if 'behavioral_features' not in self.feature_cache:
                await self.create_behavioral_features({})
            
            # Merge toutes les features
            customer_features = self.feature_cache['customer_features']
            rfm_features = self.feature_cache['rfm_features']
            behavioral_features = self.feature_cache['behavioral_features']
            
            # Join sur customer_id
            ml_dataset = customer_features.merge(rfm_features, on='customer_id', how='left')
            ml_dataset = ml_dataset.merge(behavioral_features, on='customer_id', how='left')
            
            # Création de la target variable (exemple: probabilité de churn)
            # Combinaison de plusieurs signaux faibles
            ml_dataset['target_churn'] = (
                (ml_dataset['churn_risk_score'] > 0.6).astype(int) * 0.4 +
                (ml_dataset['recency_days'] > 90).astype(int) * 0.3 +
                (ml_dataset['rfm_score'] < 2.0).astype(int) * 0.3
            )
            ml_dataset['target_churn'] = (ml_dataset['target_churn'] > 0.5).astype(int)
            
            # Sélection des features numériques pour ML
            numerical_features = [
                'age', 'days_since_registration', 'loyalty_score',
                'recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'rfm_score',
                'engagement_score', 'satisfaction_score', 'interest_diversity', 'churn_risk_score'
            ]
            
            categorical_features = ['gender_encoded', 'city_encoded', 'customer_segment']
            
            # Préparation du dataset final
            feature_columns = numerical_features + categorical_features
            ml_ready_dataset = ml_dataset[feature_columns + ['target_churn', 'customer_id']]
            
            # Normalisation des features numériques
            scaler = StandardScaler()
            ml_ready_dataset[numerical_features] = scaler.fit_transform(
                ml_ready_dataset[numerical_features].fillna(0)
            )
            self.scalers['standard_scaler'] = scaler
            
            # Encodage final des variables catégorielles
            ml_ready_dataset = pd.get_dummies(
                ml_ready_dataset, 
                columns=['customer_segment'], 
                prefix='segment'
            )
            
            self.feature_cache['ml_dataset'] = ml_ready_dataset
            
            return {
                'status': 'success',
                'message': f'ML dataset prepared with {len(ml_ready_dataset)} samples',
                'features_count': len(ml_ready_dataset.columns) - 2,  # -target -customer_id
                'target_distribution': ml_ready_dataset['target_churn'].value_counts().to_dict(),
                'data_key': 'ml_dataset',
                'feature_columns': [col for col in ml_ready_dataset.columns if col not in ['target_churn', 'customer_id']]
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error preparing ML dataset: {str(e)}'}
    

    def get_data(self, key: str) -> pd.DataFrame:

        return self.feature_cache.get(key)
    
    async def start(self):

        print(f"[{self.name}] Starting agent...")

        await self.client.connect("DataCollector", "ws://localhost:8001") #son port 
        await self.server.start()





###############################################################3
async def test_feature_engineer():

    agent = FeatureEngineerAgent()
    
    server_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n=== Test Feature Engineer Agent ===")
    
    result = await agent.create_customer_features({})
    print(f"Customer features: {result}")
    
    # Test RFM
    result = await agent.create_rfm_features({})
    print(f"RFM features: {result}")
    
    # Test behavioral
    result = await agent.create_behavioral_features({})
    print(f"Behavioral features: {result}")
    
    # Test ML dataset
    result = await agent.prepare_ml_dataset({})
    print(f"ML dataset: {result}")
    
    #   echantillon 

    ml_data = agent.get_data('ml_dataset')
    
    if ml_data is not None:
        print(f"\nML Dataset shape: {ml_data.shape}")
        print(f"Features: {ml_data.columns.tolist()}")
        print(f"\nSample data:\n{ml_data.head()}")




if __name__ == "__main__":
    asyncio.run(test_feature_engineer())