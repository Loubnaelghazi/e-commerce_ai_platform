import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json
import pickle
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import os
from src.mcp.server import MCPServer
from src.mcp.client import MCPClient

class ModelTrainerAgent:

    def __init__(self):
        self.name = "ModelTrainer"
        self.server = MCPServer(self.name, port=8003)
        self.client = MCPClient(self.name)
        self.models = {}
        self.model_performances = {}
        self.best_model = None
        
        self.server.add_tool("train_multiple_models", self.train_multiple_models)
        self.server.add_tool("hyperparameter_tuning", self.hyperparameter_tuning)
        self.server.add_tool("evaluate_model", self.evaluate_model)
        self.server.add_tool("select_best_model", self.select_best_model)
        self.server.add_tool("save_model", self.save_model)
        
    async def train_multiple_models(self, args: Dict[str, Any]) -> Dict[str, Any]:

        print(f"[{self.name}] Training multiple ML models...")
        
        try:
            #  le dataset ML depuis FeatureEngineer via MCP
            dataset_result = await self.client.call_tool(
                "FeatureEngineer", 
                "prepare_ml_dataset", 
                {}
            )
            
            if dataset_result.get('status') != 'success':
                return {'status': 'error', 'message': 'Failed to get ML dataset'}
            
            #ONLY A TEST ON A DATA qu on va simuler
            np.random.seed(42)
            n_samples = 1000
            n_features = 15
            
            X = np.random.randn(n_samples, n_features)
            # Cune target avec une logique business
            y = ((X[:, 0] > 0.5) & (X[:, 1] < -0.2) | (X[:, 2] > 1.0)).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=36, stratify=y )
            
            #  modeles 
            models_to_train = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
                'xgboost': xgb.XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
                'svm': SVC(random_state=42, probability=True)
            }
            
            training_results = {}
            
            for model_name, model in models_to_train.items():
                print(f"[{self.name}] Training {model_name}...")
                
                start_time = datetime.now()
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                training_results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time
                }
                
                # Save
                self.models[model_name] = model
                self.model_performances[model_name] = {
                    **metrics,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_time': training_time
                }
            
            return {
                'status': 'success',
                'message': f'Trained {len(models_to_train)} models successfully',
                'models_trained': list(models_to_train.keys()),
                'performance_summary': {
                    model: {
                        'accuracy': round(perf['accuracy'], 4),
                        'f1': round(perf['f1'], 4),
                        'roc_auc': round(perf['roc_auc'], 4),
                        'cv_score': round(perf['cv_mean'], 4)
                    }
                    for model, perf in self.model_performances.items()
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error training models: {str(e)}'}
    



    async def hyperparameter_tuning(self, args: Dict[str, Any]) -> Dict[str, Any]:

        print(f"[{self.name}] Performing hyperparameter tuning...")
        
        try:
            model_name = args.get('model_name', 'random_forest')
            
            if model_name not in self.models:
                return {'status': 'error', 'message': f'Model {model_name} not found. Train models first.'}
            
            np.random.seed(42)
            n_samples = 1000
            n_features = 15
            X = np.random.randn(n_samples, n_features)
            y = ((X[:, 0] > 0.5) & (X[:, 1] < -0.2) | (X[:, 2] > 1.0)).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
            
            if model_name not in param_grids:
                return {
                    'status': 'error', 
                    'message': f'Hyperparameter tuning not configured for {model_name}'
                }
            
            base_models = {
                'random_forest': RandomForestClassifier(random_state=42),
                'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'gradient_boosting': GradientBoostingClassifier(random_state=42)
            }
            
            base_model = base_models[model_name]
            param_grid = param_grids[model_name]
            
            # GridSearch avec validation croisee
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,  # 3-fold pour la demo (5-fold en production)
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            print(f"[{self.name}] Starting grid search for {model_name}...")
            start_time = datetime.now()
            
            grid_search.fit(X_train, y_train)
            
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            tuned_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            optimized_model_name = f"{model_name}_tuned"
            self.models[optimized_model_name] = best_model
            self.model_performances[optimized_model_name] = {
                **tuned_metrics,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'tuning_time': tuning_time
            }
            
            return {
                'status': 'success',
                'message': f'Hyperparameter tuning completed for {model_name}',
                'best_params': grid_search.best_params_,
                'best_cv_score': round(grid_search.best_score_, 4),
                'tuned_metrics': {k: round(v, 4) for k, v in tuned_metrics.items()},
                'improvement': {
                    'accuracy': round(tuned_metrics['accuracy'] - self.model_performances[model_name]['accuracy'], 4),
                    'roc_auc': round(tuned_metrics['roc_auc'] - self.model_performances[model_name]['roc_auc'], 4)
                },
                'tuning_time': round(tuning_time, 2),
                'optimized_model_name': optimized_model_name
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error in hyperparameter tuning: {str(e)}'}
    
    async def evaluate_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Évalue un modèle spécifique avec des métriques détaillées"""
        print(f"[{self.name}] Evaluating model...")
        
        try:
            model_name = args.get('model_name')
            if not model_name:
                return {'status': 'error', 'message': 'model_name is required'}
            
            if model_name not in self.models:
                return {'status': 'error', 'message': f'Model {model_name} not found'}
            
            model = self.models[model_name]
            
            np.random.seed(36)
            n_samples = 200  
            n_features = 15
            X_test = np.random.randn(n_samples, n_features)
            y_test = ((X_test[:, 0] > 0.5) & (X_test[:, 1] < -0.2) | (X_test[:, 2] > 1.0)).astype(int)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            detailed_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = {
                    f'feature_{i}': importance 
                    for i, importance in enumerate(model.feature_importances_)
                }
            elif hasattr(model, 'coef_'):
                feature_importance = {
                    f'feature_{i}': abs(coef) 
                    for i, coef in enumerate(model.coef_[0])
                }
            
            prediction_stats = {
                'positive_predictions': int(np.sum(y_pred)),
                'negative_predictions': int(len(y_pred) - np.sum(y_pred)),
                'positive_rate': float(np.mean(y_pred)),
                'actual_positive_rate': float(np.mean(y_test))
            }
            
            return {
                'status': 'success',
                'message': f'Model {model_name} evaluated successfully',
                'model_name': model_name,
                'metrics': {k: round(v, 4) for k, v in detailed_metrics.items()},
                'confusion_matrix': {
                    'true_negative': int(cm[0, 0]),
                    'false_positive': int(cm[0, 1]),
                    'false_negative': int(cm[1, 0]),
                    'true_positive': int(cm[1, 1])
                },
                'feature_importance': feature_importance,
                'prediction_stats': prediction_stats,
                'test_samples': len(y_test)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error evaluating model: {str(e)}'}
    
    async def select_best_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sélectionne le meilleur modèle basé sur les métriques"""
        print(f"[{self.name}] Selecting best model...")
        
        try:
            if not self.model_performances:
                return {'status': 'error', 'message': 'No models trained yet'}
            
            metric = args.get('metric', 'roc_auc')  
            
            best_model_name = max(
                self.model_performances.keys(),
                key=lambda x: self.model_performances[x].get(metric, 0)
            )
            
            best_performance = self.model_performances[best_model_name]
            self.best_model = {
                'name': best_model_name,
                'model': self.models[best_model_name],
                'performance': best_performance
            }
            
            model_comparison = {}
            for model_name, performance in self.model_performances.items():
                model_comparison[model_name] = {
                    'accuracy': round(performance.get('accuracy', 0), 4),
                    'f1': round(performance.get('f1', 0), 4),
                    'roc_auc': round(performance.get('roc_auc', 0), 4),
                    'is_best': model_name == best_model_name
                }
            
            return {
                'status': 'success',
                'message': f'Best model selected: {best_model_name}',
                'best_model': best_model_name,
                'selection_metric': metric,
                'best_score': round(best_performance.get(metric, 0), 4),
                'model_comparison': model_comparison,
                'performance_summary': {
                    'accuracy': round(best_performance.get('accuracy', 0), 4),
                    'precision': round(best_performance.get('precision', 0), 4),
                    'recall': round(best_performance.get('recall', 0), 4),
                    'f1': round(best_performance.get('f1', 0), 4),
                    'roc_auc': round(best_performance.get('roc_auc', 0), 4)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error selecting best model: {str(e)}'}
    
    async def save_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Sauvegarde un modèle sur disque"""
        print(f"[{self.name}] Saving model...")
        
        try:
            model_name = args.get('model_name')
            if not model_name:
                if self.best_model:
                    model_name = self.best_model['name']
                else:
                    return {'status': 'error', 'message': 'No model specified and no best model selected'}
            
            if model_name not in self.models:
                return {'status': 'error', 'message': f'Model {model_name} not found'}
            
            model = self.models[model_name]
            save_path = args.get('save_path', f'../../data/models/{model_name}.pkl')
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Sauvegarde avec pickle
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'performance': self.model_performances.get(model_name, {}),
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name
                }, f)
            
            metadata_path = save_path.replace('.pkl', '_metadata.json')
            metadata = {
                'model_name': model_name,
                'performance': self.model_performances.get(model_name, {}),
                'timestamp': datetime.now().isoformat(),
                'file_path': save_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'status': 'success',
                'message': f'Model {model_name} saved successfully',
                'model_name': model_name,
                'save_path': save_path,
                'metadata_path': metadata_path,
                'model_size': os.path.getsize(save_path),
                'performance': self.model_performances.get(model_name, {})
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error saving model: {str(e)}'}
    





    def get_model(self, model_name: str):
        return self.models.get(model_name)
    
    def get_best_model(self):
        return self.best_model
    
    async def start(self):

        print(f"[{self.name}] Starting agent...")
        await self.client.connect("FeatureEngineer", "ws://localhost:8002")
        await self.server.start()

async def test_model_trainer():
    agent = ModelTrainerAgent()
    
    server_task = asyncio.create_task(agent.start())
    await asyncio.sleep(1)
    
    print("\n=== Test Model Trainer Agent ===")
    
    result = await agent.train_multiple_models({})
    print(f"Multiple models training: {result}")
    
    result = await agent.hyperparameter_tuning({'model_name': 'random_forest'})
    print(f"Hyperparameter tuning: {result}")
    
    result = await agent.evaluate_model({'model_name': 'random_forest'})
    print(f"Model evaluation: {result}")
    
    result = await agent.select_best_model({'metric': 'roc_auc'})
    print(f"Best model selection: {result}")
    
    result = await agent.save_model({})
    print(f"Model saving: {result}")

if __name__ == "__main__":
    asyncio.run(test_model_trainer())