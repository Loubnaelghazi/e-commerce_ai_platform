import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import json
import pickle
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score

from src.mcp.server import MCPServer
from src.mcp.client import MCPClient

class PredictorAgent:

    def __init__(self):

        self.name = "Predictor"
        self.server = MCPServer(self.name, port=8004)
        self.client = MCPClient(self.name)
        self.loaded_models = {}
        self.prediction_history = []
        self.model_performance_monitor = {}
        
        self.server.add_tool("load_model", self.load_model)
        self.server.add_tool("predict_single", self.predict_single)
        self.server.add_tool("predict_batch", self.predict_batch)
        self.server.add_tool("predict_realtime", self.predict_realtime)
        self.server.add_tool("monitor_model_drift", self.monitor_model_drift)
        
    async def load_model(self, args: Dict[str, Any]) -> Dict[str, Any]:

        print(f"[{self.name}] Loading model...")
        
        try:
            model_name = args.get('model_name')
            load_from = args.get('load_from', 'trainer') 
            
            if load_from == 'trainer':
                # Charge le meilleur modèle depuis ModelTrainer via MCP
                best_model_result = await self.client.call_tool(
                    "ModelTrainer", 
                    "select_best_model", 
                    {}
                )
                
                if best_model_result.get('status') != 'success':
                    return {'status': 'error', 'message': 'Failed to get best model from trainer'}
                
                # Pour la démo, on simule le chargement du modèle
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(random_state=42, n_estimators=100)
                
                # Entraînement rapide pour la démo
                np.random.seed(42)
                X_demo = np.random.randn(1000, 15)
                y_demo = ((X_demo[:, 0] > 0.5) & (X_demo[:, 1] < -0.2) | (X_demo[:, 2] > 1.0)).astype(int)
                model.fit(X_demo, y_demo)
                
                model_info = {
                    'model': model,
                    'name': best_model_result.get('best_model', 'random_forest'),
                    'performance': best_model_result.get('performance_summary', {}),
                    'loaded_at': datetime.now().isoformat(),
                    'source': 'trainer'
                }
                
            else:  # load_from == 'disk'
                model_path = args.get('model_path', f'../../data/models/{model_name}.pkl')
                
                try:
                    with open(model_path, 'rb') as f:
                        saved_data = pickle.load(f)
                    
                    model_info = {
                        'model': saved_data['model'],
                        'name': saved_data['model_name'],
                        'performance': saved_data['performance'],
                        'loaded_at': datetime.now().isoformat(),
                        'source': 'disk',
                        'original_timestamp': saved_data['timestamp']
                    }
                    
                except FileNotFoundError:
                    return {'status': 'error', 'message': f'Model file not found: {model_path}'}
            
            # Sauvegarde du modèle chargé
            self.loaded_models[model_info['name']] = model_info
            
            return {
                'status': 'success',
                'message': f'Model {model_info["name"]} loaded successfully',
                'model_name': model_info['name'],
                'source': model_info['source'],
                'performance': model_info['performance'],
                'loaded_at': model_info['loaded_at']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error loading model: {str(e)}'}
    
    async def predict_single(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fait une prédiction pour un seul échantillon"""
        print(f"[{self.name}] Making single prediction...")
        
        try:
            model_name = args.get('model_name')
            features = args.get('features')  # Liste ou array des features
            
            if not model_name:
                # Utilise le premier modèle disponible
                if not self.loaded_models:
                    return {'status': 'error', 'message': 'No models loaded'}
                model_name = list(self.loaded_models.keys())[0]
            
            if model_name not in self.loaded_models:
                return {'status': 'error', 'message': f'Model {model_name} not loaded'}
            
            if not features:
                return {'status': 'error', 'message': 'Features are required'}
            
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            
            # Conversion des features en numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Prédiction
            prediction = model.predict(features_array)[0]
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features_array)[0]
            
            # Enregistrement de la prédiction
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'features': features,
                'prediction': int(prediction),
                'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
                'type': 'single'
            }
            
            self.prediction_history.append(prediction_record)
            
            return {
                'status': 'success',
                'message': 'Single prediction completed',
                'model_name': model_name,
                'prediction': int(prediction),
                'prediction_proba': prediction_proba.tolist() if prediction_proba is not None else None,
                'confidence': float(max(prediction_proba)) if prediction_proba is not None else None,
                'timestamp': prediction_record['timestamp']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error making prediction: {str(e)}'}
    
    async def predict_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fait des prédictions pour un batch d'échantillons"""
        print(f"[{self.name}] Making batch predictions...")
        
        try:
            model_name = args.get('model_name')
            features_batch = args.get('features_batch')  # List de listes ou array 2D
            batch_size = args.get('batch_size', 100)
            
            if not model_name:
                if not self.loaded_models:
                    return {'status': 'error', 'message': 'No models loaded'}
                model_name = list(self.loaded_models.keys())[0]
            
            if model_name not in self.loaded_models:
                return {'status': 'error', 'message': f'Model {model_name} not loaded'}
            
            if not features_batch:
                return {'status': 'error', 'message': 'Features batch is required'}
            
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            
            # Conversion en numpy array
            features_array = np.array(features_batch)
            
            # Prédictions par batch pour optimiser la performance
            predictions = []
            predictions_proba = []
            
            for i in range(0, len(features_array), batch_size):
                batch = features_array[i:i+batch_size]
                
                batch_pred = model.predict(batch)
                predictions.extend(batch_pred)
                
                if hasattr(model, 'predict_proba'):
                    batch_proba = model.predict_proba(batch)
                    predictions_proba.extend(batch_proba)
            
            # Statistiques des prédictions
            predictions_array = np.array(predictions)
            positive_predictions = np.sum(predictions_array)
            confidence_scores = [max(proba) for proba in predictions_proba] if predictions_proba else []
            
            # Enregistrement
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'batch_size': len(features_array),
                'positive_predictions': int(positive_predictions),
                'negative_predictions': int(len(predictions) - positive_predictions),
                'avg_confidence': float(np.mean(confidence_scores)) if confidence_scores else None,
                'type': 'batch'
            }
            
            self.prediction_history.append(prediction_record)
            
            return {
                'status': 'success',
                'message': f'Batch prediction completed for {len(features_array)} samples',
                'model_name': model_name,
                'total_samples': len(features_array),
                'predictions': [int(p) for p in predictions],
                'predictions_proba': [p.tolist() for p in predictions_proba] if predictions_proba else None,
                'summary': {
                    'positive_predictions': int(positive_predictions),
                    'negative_predictions': int(len(predictions) - positive_predictions),
                    'positive_rate': float(positive_predictions / len(predictions)),
                    'avg_confidence': float(np.mean(confidence_scores)) if confidence_scores else None
                },
                'timestamp': prediction_record['timestamp']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error making batch predictions: {str(e)}'}
    
    async def predict_realtime(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Simule des prédictions en temps réel avec streaming de données"""
        print(f"[{self.name}] Starting realtime predictions...")
        
        try:
            model_name = args.get('model_name')
            duration_seconds = args.get('duration_seconds', 30)
            predictions_per_second = args.get('predictions_per_second', 5)
            
            if not model_name:
                if not self.loaded_models:
                    return {'status': 'error', 'message': 'No models loaded'}
                model_name = list(self.loaded_models.keys())[0]
            
            if model_name not in self.loaded_models:
                return {'status': 'error', 'message': f'Model {model_name} not loaded'}
            
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            
            realtime_predictions = []
            total_predictions = 0
            start_time = datetime.now()
            
            print(f"[{self.name}] Running realtime predictions for {duration_seconds}s...")
            
            # Simulation du streaming
            for second in range(duration_seconds):
                for _ in range(predictions_per_second):
                    # Génération de données aléatoires (simulation streaming)
                    features = np.random.randn(15)  # 15 features
                    
                    # Prédiction
                    prediction = model.predict(features.reshape(1, -1))[0]
                    prediction_proba = None
                    
                    if hasattr(model, 'predict_proba'):
                        prediction_proba = model.predict_proba(features.reshape(1, -1))[0]
                    
                    realtime_predictions.append({
                        'timestamp': datetime.now().isoformat(),
                        'prediction': int(prediction),
                        'confidence': float(max(prediction_proba)) if prediction_proba is not None else None
                    })
                    
                    total_predictions += 1
                
                # Pause d'une seconde
                await asyncio.sleep(1)
            
            end_time = datetime.now()
            actual_duration = (end_time - start_time).total_seconds()
            
            # Statistiques
            positive_predictions = sum(1 for p in realtime_predictions if p['prediction'] == 1)
            avg_confidence = np.mean([p['confidence'] for p in realtime_predictions if p['confidence']])
            
            # Enregistrement
            prediction_record = {
                'timestamp': start_time.isoformat(),
                'model_name': model_name,
                'duration_seconds': actual_duration,
                'total_predictions': total_predictions,
                'predictions_per_second': total_predictions / actual_duration,
                'positive_rate': positive_predictions / total_predictions,
                'avg_confidence': float(avg_confidence),
                'type': 'realtime'
            }
            
            self.prediction_history.append(prediction_record)
            
            return {
                'status': 'success',
                'message': f'Realtime predictions completed: {total_predictions} predictions in {actual_duration:.1f}s',
                'model_name': model_name,
                'duration_seconds': actual_duration,
                'total_predictions': total_predictions,
                'predictions_per_second': round(total_predictions / actual_duration, 2),
                'summary': {
                    'positive_predictions': positive_predictions,
                    'negative_predictions': total_predictions - positive_predictions,
                    'positive_rate': round(positive_predictions / total_predictions, 4),
                    'avg_confidence': round(float(avg_confidence), 4)
                },
                'sample_predictions': realtime_predictions[:10]  # Échantillon des 10 premières
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error in realtime predictions: {str(e)}'}
    
    async def monitor_model_drift(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Surveille la dérive du modèle en comparant les performances récentes"""
        print(f"[{self.name}] Monitoring model drift...")
        
        try:
            model_name = args.get('model_name')
            window_size = args.get('window_size', 100)  # Nombre de prédictions à analyser
            
            if not model_name:
                if not self.loaded_models:
                    return {'status': 'error', 'message': 'No models loaded'}
                model_name = list(self.loaded_models.keys())[0]
            
            if model_name not in self.loaded_models:
                return {'status': 'error', 'message': f'Model {model_name} not loaded'}
            
            # Filtre l'historique pour le modèle spécifique
            model_predictions = [
                p for p in self.prediction_history 
                if p.get('model_name') == model_name
            ]
            
            if len(model_predictions) < window_size:
                return {
                    'status': 'warning',
                    'message': f'Insufficient predictions for drift analysis: {len(model_predictions)} < {window_size}',
                    'available_predictions': len(model_predictions)
                }
            
            # Analyse des prédictions récentes vs historiques
            recent_predictions = model_predictions[-window_size:]
            historical_predictions = model_predictions[:-window_size] if len(model_predictions) > window_size else []
            
            # Calcul de la dérive
            recent_positive_rate = np.mean([
                p.get('prediction', 0) for p in recent_predictions 
                if 'prediction' in p
            ])
            
            recent_avg_confidence = np.mean([
                p.get('confidence', 0) for p in recent_predictions 
                if p.get('confidence') is not None
            ])
            
            drift_metrics = {
                'recent_positive_rate': float(recent_positive_rate),
                'recent_avg_confidence': float(recent_avg_confidence),
                'drift_detected': False,
                'drift_severity': 'low'
            }
            
            if historical_predictions:
                historical_positive_rate = np.mean([
                    p.get('prediction', 0) for p in historical_predictions 
                    if 'prediction' in p
                ])
                
                historical_avg_confidence = np.mean([
                    p.get('confidence', 0) for p in historical_predictions 
                    if p.get('confidence') is not None
                ])
                
                # Détection de dérive
                positive_rate_drift = abs(recent_positive_rate - historical_positive_rate)
                confidence_drift = abs(recent_avg_confidence - historical_avg_confidence)
                
                drift_metrics.update({
                    'historical_positive_rate': float(historical_positive_rate),
                    'historical_avg_confidence': float(historical_avg_confidence),
                    'positive_rate_drift': float(positive_rate_drift),
                    'confidence_drift': float(confidence_drift)
                })
                
                # Seuils de détection
                if positive_rate_drift > 0.1 or confidence_drift > 0.1:
                    drift_metrics['drift_detected'] = True
                    drift_metrics['drift_severity'] = 'high' if positive_rate_drift > 0.2 else 'medium'
            
            # Recommandations
            recommendations = []
            if drift_metrics['drift_detected']:
                recommendations.extend([
                    "Model drift detected - consider retraining",
                    "Investigate data quality changes",
                    "Review feature distribution shifts"
                ])
            
            if drift_metrics['recent_avg_confidence'] < 0.7:
                recommendations.append("Low confidence scores - consider model ensemble")
            
            # Mise à jour du monitoring
            self.model_performance_monitor[model_name] = {
                'last_check': datetime.now().isoformat(),
                'drift_metrics': drift_metrics,
                'recommendations': recommendations
            }
            
            return {
                'status': 'success',
                'message': f'Model drift analysis completed for {model_name}',
                'model_name': model_name,
                'window_size': window_size,
                'total_predictions_analyzed': len(model_predictions),
                'drift_metrics': drift_metrics,
                'recommendations': recommendations,
                'requires_attention': drift_metrics['drift_detected']
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error monitoring model drift: {str(e)}'}
    
    def get_prediction_history(self):
        """Récupère l'historique des prédictions"""
        return self.prediction_history
    
    def get_loaded_models(self):
        """Récupère la liste des modèles chargés"""
        return list(self.loaded_models.keys())
    
    async def start(self):
        """Démarre l'agent"""
        print(f"[{self.name}] Starting agent...")
        # Connexion au ModelTrainer
        await self.client.connect("ModelTrainer", "ws://localhost:8003")
        await self.server.start()

# Test de l'agent
async def test_predictor():
    agent = PredictorAgent()
    
    # Démarre le serveur
    server_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n=== Test Predictor Agent ===")
    
    # Test chargement de modèle
    result = await agent.load_model({'load_from': 'trainer'})
    print(f"Model loading: {result}")
    
    # Test prédiction simple
    sample_features = np.random.randn(15).tolist()  # 15 features aléatoires
    result = await agent.predict_single({'features': sample_features})
    print(f"Single prediction: {result}")
    
    # Test prédiction batch
    batch_features = [np.random.randn(15).tolist() for _ in range(20)]
    result = await agent.predict_batch({'features_batch': batch_features})
    print(f"Batch predictions: {result}")
    
    # Test prédictions temps réel (courte durée pour la démo)
    result = await agent.predict_realtime({'duration_seconds': 5, 'predictions_per_second': 3})
    print(f"Realtime predictions: {result}")
    
    # Test monitoring de dérive
    result = await agent.monitor_model_drift({})
    print(f"Model drift monitoring: {result}")
    
    # Affichage de l'historique
    history = agent.get_prediction_history()
    print(f"\nPrediction history: {len(history)} records")
    for record in history[-3:]:  # Derniers 3 enregistrements
        print(f"  {record['type']}: {record['timestamp']}")

if __name__ == "__main__":
    asyncio.run(test_predictor())