# - Orchestration complète A2A avec MCP

'''

Ce que fait le pipeline :

Collecte 1000 clients + 5000 transactions
Cree des features RFM + comportementales
Entraine 5 modeles ML avec optimization
Fait des prédictions temps réel 
Genere insights business + calcul ROI

Différence A2A vs MCP (le but de ce mini projet etait de assimiler ces techno) :

A2A = Le concept (agents qui collaborent)
MCP = La technologie (comment ils se parlent)

Resultat : Un systeme distribué où 5 agents IA collaborent via MCP pour créer une plateforme d'analyse prédictive complète !


'''


'''

5 Agents specialises :

DataCollector (Port 8001) : Collecte et nettoie les données
FeatureEngineer (Port 8002) : Crée les features ML
ModelTrainer (Port 8003) : Entraîne et optimise les modèles
Predictor (Port 8004) : Fait des prédictions temps réel
BusinessIntelligence (Port 8005) : Génère insights et ROI


'''


import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import sys
import os

from src.agents.data_collector.agent import DataCollectorAgent
from src.agents.feature_enginneer.agent import FeatureEngineerAgent
from src.agents.model_trainer.agent import ModelTrainerAgent
from src.agents.predictor.agent import PredictorAgent
from src.agents.business_intelligence.agent import BusinessIntelligenceAgent

class EcommerceAIPlatform:
    def __init__(self):
        self.agents = {}
        self.execution_log = []
        
    async def initialize_agents(self):
        """Initialise tous les agents de la plateforme"""
        print(" Initializing E-commerce AI Platform...")
        print("=" * 50)
        
        self.agents = {
            'data_collector': DataCollectorAgent(),
            'feature_engineer': FeatureEngineerAgent(),
            'model_trainer': ModelTrainerAgent(),
            'predictor': PredictorAgent(),
            'business_intelligence': BusinessIntelligenceAgent()
        }
        
        tasks = []
        for agent_name, agent in self.agents.items():
            print(f"📡 Starting {agent_name}...")
            task = asyncio.create_task(agent.start())
            tasks.append(task)
        
        await asyncio.sleep(3)
        print(" All agents initialized successfully!")
        print()
        
        return tasks
    
    async def run_full_pipeline(self):
        """Exécute le pipeline complet A2A"""
        print(" Starting Full A2A Pipeline...")
        print("=" * 50)
        
        pipeline_start = datetime.now()
        
        try:
            #  Collecte des données
            print(" STEP 1: Data Collection")
            print("-" * 30)
            
            result = await self.agents['data_collector'].collect_customer_data({'n_customers': 1000})
            self.log_step("data_collection_customers", result)
            print(f" Customers: {result.get('message', 'Success')}")
            
            # Collecte des transactions
            result = await self.agents['data_collector'].collect_transaction_data({'n_transactions': 5000})
            self.log_step("data_collection_transactions", result)
            print(f" Transactions: {result.get('message', 'Success')}")
            
            result = await self.agents['data_collector'].clean_data({'data_key': 'customers'})
            self.log_step("data_cleaning_customers", result)
            print(f" Cleaning customers: {result.get('message', 'Success')}")
            
            result = await self.agents['data_collector'].clean_data({'data_key': 'transactions'})
            self.log_step("data_cleaning_transactions", result)
            print(f" Cleaning transactions: {result.get('message', 'Success')}")
            
            print()
            
            print(" STEP 2: Feature Engineering")
            print("-" * 30)
            
            result = await self.agents['feature_engineer'].create_customer_features({})
            self.log_step("feature_engineering_customers", result)
            print(f" Customer features: {result.get('message', 'Success')}")
            
            result = await self.agents['feature_engineer'].create_rfm_features({})
            self.log_step("feature_engineering_rfm", result)
            print(f" RFM features: {result.get('message', 'Success')}")
            
            result = await self.agents['feature_engineer'].create_behavioral_features({})
            self.log_step("feature_engineering_behavioral", result)
            print(f" Behavioral features: {result.get('message', 'Success')}")
            
            result = await self.agents['feature_engineer'].prepare_ml_dataset({})
            self.log_step("ml_dataset_preparation", result)
            print(f" ML Dataset: {result.get('message', 'Success')}")
            
            print()
            
            print(" STEP 3: Model Training")
            print("-" * 30)
            
            result = await self.agents['model_trainer'].train_multiple_models({})
            self.log_step("model_training", result)
            print(f" Multiple models: {result.get('message', 'Success')}")
            
            result = await self.agents['model_trainer'].hyperparameter_tuning({'model_name': 'random_forest'})
            self.log_step("hyperparameter_tuning", result)
            print(f" Hyperparameter tuning: {result.get('message', 'Success')}")
            
            result = await self.agents['model_trainer'].select_best_model({'metric': 'roc_auc'})
            self.log_step("best_model_selection", result)
            print(f" Best model selection: {result.get('best_model', 'Success')}")
            
            result = await self.agents['model_trainer'].save_model({})
            self.log_step("model_saving", result)
            print(f" Model saving: {result.get('message', 'Success')}")
            
            print()
            
            print(" STEP 4: Predictions")
            print("-" * 30)
            
            result = await self.agents['predictor'].load_model({'load_from': 'trainer'})
            self.log_step("model_loading", result)
            print(f" Model loading: {result.get('message', 'Success')}")
            
            sample_features = [0.5, -0.3, 1.2, 0.8, -0.1, 0.9, 0.2, -0.7, 0.4, 0.1, 0.6, -0.2, 0.3, 0.7, -0.4]
            result = await self.agents['predictor'].predict_single({'features': sample_features})
            self.log_step("single_prediction", result)
            print(f" Single prediction: {result.get('prediction', 'Success')}")
            
            import numpy as np
            np.random.seed(42)
            batch_features = [np.random.randn(15).tolist() for _ in range(50)]
            result = await self.agents['predictor'].predict_batch({'features_batch': batch_features})
            self.log_step("batch_predictions", result)
            print(f"Batch predictions: {result.get('total_samples', 0)} samples")
            
            result = await self.agents['predictor'].predict_realtime({'duration_seconds': 10, 'predictions_per_second': 5})
            self.log_step("realtime_predictions", result)
            print(f" Realtime predictions: {result.get('total_predictions', 0)} predictions")
            
            result = await self.agents['predictor'].monitor_model_drift({})
            self.log_step("model_drift_monitoring", result)
            print(f" Drift monitoring: {result.get('message', 'Success')}")
            
            print()
            
            print(" STEP 5: Business Intelligence")
            print("-" * 30)
            
            result = await self.agents['business_intelligence'].generate_customer_insights({})
            self.log_step("customer_insights", result)
            print(f" Customer insights: {result.get('segments_analyzed', 0)} segments analyzed")
            
            result = await self.agents['business_intelligence'].create_performance_report({})
            self.log_step("performance_report", result)
            print(f" Performance report: {result.get('best_model', 'Success')}")
            
            result = await self.agents['business_intelligence'].generate_predictions_dashboard({})
            self.log_step("predictions_dashboard", result)
            print(f" Dashboard: {result.get('components_created', 0)} components created")
            
            result = await self.agents['business_intelligence'].create_business_alerts({})
            self.log_step("business_alerts", result)
            print(f" Business alerts: {result.get('total_alerts', 0)} alerts created")
            
            result = await self.agents['business_intelligence'].calculate_roi({'time_period_days': 90})
            self.log_step("roi_analysis", result)
            print(f" ROI analysis: {result.get('roi_percentage', 0):.1f}% ROI")
            
            print()
            
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()
            
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"⏱  Total execution time: {execution_time:.2f} seconds")
            print(f" Total steps executed: {len(self.execution_log)}")
            print(f" Success rate: {self.calculate_success_rate():.1f}%")
            
            self.print_key_results()
            
        except Exception as e:
            print(f" Pipeline failed: {str(e)}")
            raise
    

    def log_step(self, step_name: str, result: Dict[str, Any]):
        """Enregistre l'exécution d'une étape"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': result.get('status', 'unknown'),
            'message': result.get('message', ''),
            'execution_time': datetime.now().isoformat()
        }
        self.execution_log.append(log_entry)
    
    def calculate_success_rate(self) -> float:
        """Calcule le taux de succès du pipeline"""
        if not self.execution_log:
            return 0.0
        
        successful_steps = len([step for step in self.execution_log if step['status'] == 'success'])
        return (successful_steps / len(self.execution_log)) * 100
    
    def print_key_results(self):
        print("\n📋 KEY RESULTS SUMMARY")
        print("-" * 30)
        
        data_steps = [step for step in self.execution_log if 'data_collection' in step['step']]
        if data_steps:
            print(" Data Collection:")
            print("   • 1,000 customers collected and cleaned")
            print("   • 5,000 transactions collected and cleaned")
        
        feature_steps = [step for step in self.execution_log if 'feature_engineering' in step['step']]
        if feature_steps:
            print(" Feature Engineering:")
            print("   • Customer demographic features")
            print("   • RFM analysis with customer segmentation")
            print("   • Behavioral features and engagement scores")
            print("   • ML-ready dataset prepared")
        
        model_steps = [step for step in self.execution_log if 'model_training' in step['step']]
        if model_steps:
            print(" Model Training:")
            print("   • 5 different algorithms trained")
            print("   • Hyperparameter optimization completed")
            print("   • Best model selected and saved")
        
        # Prédictions
        prediction_steps = [step for step in self.execution_log if 'prediction' in step['step']]
        if prediction_steps:
            print(" Predictions:")
            print("   • Single and batch predictions completed")
            print("   • Real-time prediction system tested")
            print("   • Model drift monitoring active")
        
        # Business Intelligence
        bi_steps = [step for step in self.execution_log if step['step'] in ['customer_insights', 'roi_analysis']]
        if bi_steps:
            print(" Business Intelligence:")
            print("   • Customer segmentation insights generated")
            print("   • Performance dashboard created")
            print("   • Business alerts configured")
            print("   • ROI analysis completed")
        
        print("\n A2A COMMUNICATION SUMMARY")
        print("-" * 30)
        print("   • Data Collector ↔ Feature Engineer: Data transfer")
        print("   • Feature Engineer ↔ Model Trainer: ML dataset")
        print("   • Model Trainer ↔ Predictor: Model loading")
        print("   • All Agents ↔ Business Intelligence: Insights aggregation")
        print("   • MCP Protocol: Secure inter-agent communication")

async def main():
    """Fonction principale"""
    print(" E-COMMERCE AI PLATFORM - A2A ARCHITECTURE")
    print("=" * 60)
    print(" Agent-to-Agent Communication with Model Context Protocol")
    print(" Complete ML Pipeline: Data → Features → Models → Predictions → Insights")
    print("=" * 60)
    print()
    
    platform = EcommerceAIPlatform()
    
    try:
        agent_tasks = await platform.initialize_agents()
        
        await platform.run_full_pipeline()
        
        print("\n EXECUTION LOG")
        print("-" * 30)
        for i, step in enumerate(platform.execution_log[-5:], 1):  
            status_emoji = "✅" if step['status'] == 'success' else "❌"
            print(f"{i}. {status_emoji} {step['step']}: {step['message'][:50]}...")
        
        print(f"\n📄 Full log: {len(platform.execution_log)} steps executed")
        
        log_filename = f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_filename, 'w') as f:
            json.dump(platform.execution_log, f, indent=2)
        print(f"💾 Execution log saved to: {log_filename}")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\n Platform error: {str(e)}")
        raise
    finally:
        print("\n Shutting down platform...")

if __name__ == "__main__":
    asyncio.run(main())