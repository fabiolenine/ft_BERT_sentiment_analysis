import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    get_linear_schedule_with_warmup
)

# Data loading library
from datasets import load_dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
import logging
from datetime import datetime

# Configure logging with incremental file in ../logs/train/
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'train')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp for this execution
    log_filename = f"bert_sentiment_training.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Configure logging with incremental file (append mode)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='a', encoding='utf-8'),  # Append mode for incremental logging
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== NOVA SESSÃO DE TREINAMENTO INICIADA ===")
    logger.info(f"Log sendo salvo em: {log_filepath}")
    return logger

# Setup logging
logger = setup_logging()

logger.info(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU disponível: {device_name}")
    # Limpa o cache da GPU
    torch.cuda.empty_cache()
    logger.debug("Cache da GPU limpo")
else:
    logger.warning("GPU não disponível. Usando CPU.")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"SentimentDataset criado com {len(texts)} amostras, max_length={max_length}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Preprocessamento: conversão para minúsculas
        text = text.lower()
        
        # Log de debug para alguns exemplos (evitar spam de logs)
        if idx < 5:
            logger.debug(f"Processando amostra {idx}: texto (primeiros 100 chars): {text[:100]}...")
            logger.debug(f"Amostra {idx} - label: {label}")
        
        # Tokenização com padding, truncamento e máscara de atenção
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTSentimentTrainer:
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased', max_length=128, num_labels=2
                 , dataset_name="ruanchaves/b2w-reviews01"):       
        logger.info(f"Inicializando BERTSentimentTrainer com modelo: {model_name}")
        
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Dispositivo selecionado: {self.device}")
        logger.info(f"Configurações: max_length={max_length}, num_labels={num_labels}")
        
        # Inicializar tokenizer e modelo, carregando todos os pesos pré-treinados
        logger.info(f"Carregando tokenizer do modelo: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        logger.info(f"Carregando modelo BERT para classificação de sequência")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        logger.info(f"Modelo transferido para: {self.device}")
        
        # Métricas de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Inicialização completa - Dataset configurado: {self.dataset_name}")
        logger.info(f"Número de parâmetros do modelo: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data(self):
        """Carrega e processa o dataset B2W Reviews"""
        logger.info(f"Iniciando carregamento do dataset: {self.dataset_name}")
        
        try:
            # Carregar dataset do Hugging Face
            logger.info("Baixando dataset do Hugging Face...")
            dataset = load_dataset(self.dataset_name)
            logger.info("Dataset carregado com sucesso")
            
            # Converter para DataFrame (só tem split train)
            logger.info("Convertendo dataset para DataFrame")
            all_data = pd.DataFrame(dataset['train'])
            logger.info(f"DataFrame criado com shape: {all_data.shape}")
            
            # Log das colunas disponíveis
            logger.debug(f"Colunas disponíveis: {list(all_data.columns)}")
            
            # Assumindo que as colunas são 'text' e 'label'
            texts = all_data['review_text'].tolist()
            labels = all_data['overall_rating'].tolist()
            logger.info(f"Extraídas {len(texts)} reviews com ratings")
            
            # Converter ratings para classificação binária (1-2: negativo=0, 3-5: positivo=1)
            logger.info("Convertendo ratings para classificação binária (1-2: negativo=0, 3-5: positivo=1)")
            binary_labels = [1 if rating >= 3 else 0 for rating in labels]
            
            # Log das estatísticas
            label_counts = pd.Series(binary_labels).value_counts()
            logger.info(f"Total de amostras: {len(texts)}")
            logger.info(f"Distribuição de classes: {dict(label_counts)}")
            logger.info(f"Proporção positiva: {label_counts.get(1, 0) / len(binary_labels):.3f}")
            logger.info(f"Proporção negativa: {label_counts.get(0, 0) / len(binary_labels):.3f}")
            
            # Verificar dados válidos
            null_texts = sum(1 for text in texts if not text or pd.isna(text))
            if null_texts > 0:
                logger.warning(f"Encontrados {null_texts} textos vazios ou nulos")
            
            logger.info("Carregamento de dados concluído com sucesso")
            return texts, binary_labels
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    
    def prepare_data(self, texts, labels, test_size=0.2, val_size=0.1, 
                     seed_random_state=None):
        """Prepara os dados para treinamento"""
        logger.info(f"Iniciando preparação dos dados com test_size={test_size}, val_size={val_size}")
        logger.info(f"Random state: {seed_random_state}")
        
        try:
            # Primeiro, separa o teste
            logger.info("Separando conjunto de teste")
            X_temp, X_test, y_temp, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=seed_random_state, stratify=labels)
            logger.info(f"Conjunto de teste: {len(X_test)} amostras ({test_size*100:.1f}%)")
        
            # Agora calcula o percentual real de validação sobre o restante
            val_relative = val_size / (1 - test_size)
            logger.info(f"Separando conjunto de validação (percentual relativo: {val_relative:.3f})")
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative, random_state=seed_random_state, stratify=y_temp)
            
            logger.info(f"Conjunto de treino: {len(X_train)} amostras")
            logger.info(f"Conjunto de validação: {len(X_val)} amostras")
            
            # Log das distribuições de classe em cada conjunto
            train_dist = pd.Series(y_train).value_counts()
            val_dist = pd.Series(y_val).value_counts()
            test_dist = pd.Series(y_test).value_counts()
            
            logger.info(f"Distribuição treino: {dict(train_dist)}")
            logger.info(f"Distribuição validação: {dict(val_dist)}")
            logger.info(f"Distribuição teste: {dict(test_dist)}")

            # Criar datasets
            logger.info("Criando datasets tokenizados")
            train_dataset = SentimentDataset(X_train, y_train, self.tokenizer, self.max_length)
            val_dataset = SentimentDataset(X_val, y_val, self.tokenizer, self.max_length)
            test_dataset = SentimentDataset(X_test, y_test, self.tokenizer, self.max_length)
            
            logger.info("Preparação de dados concluída com sucesso")
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Erro durante preparação dos dados: {str(e)}")
            raise

    
    def train(self, train_dataset, val_dataset, batch_size=16, epochs=3, learning_rate=2e-5):
        """Treina o modelo BERT"""
        logger.info("=== INICIANDO TREINAMENTO ===")
        logger.info(f"Parâmetros de treinamento:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Épocas: {epochs}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Tamanho do dataset de treino: {len(train_dataset)}")
        logger.info(f"  - Tamanho do dataset de validação: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"  - Número de batches de treino por época: {len(train_loader)}")
        logger.info(f"  - Número de batches de validação: {len(val_loader)}")
        
        # Otimizador e scheduler
        logger.info("Configurando otimizador e scheduler")
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        logger.info(f"  - Total de steps de treinamento: {total_steps}")
        logger.info(f"  - Warmup steps: 0")
        
        for epoch in range(epochs):
            logger.info(f"\n=== ÉPOCA {epoch + 1}/{epochs} ===")
            epoch_start_time = datetime.now()
            
            # Treinamento
            logger.info("Iniciando fase de treinamento")
            self.model.train()
            total_train_loss = 0
            train_predictions = []
            train_true_labels = []
            
            train_pbar = tqdm(train_loader, desc=f"Treinamento Época {epoch + 1}")
            batch_count = 0
            
            for batch in train_pbar:
                batch_count += 1
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Log detalhado a cada 100 batches
                if batch_count % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.debug(f"Batch {batch_count}/{len(train_loader)} - Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Métricas
                predictions = torch.argmax(outputs.logits, dim=-1)
                train_predictions.extend(predictions.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = accuracy_score(train_true_labels, train_predictions)
            
            logger.info(f"Treinamento época {epoch + 1} concluído:")
            logger.info(f"  - Loss médio: {avg_train_loss:.4f}")
            logger.info(f"  - Acurácia: {train_accuracy:.4f}")
            
            # Validação
            logger.info("Iniciando validação")
            val_loss, val_accuracy, val_metrics = self.evaluate(val_loader)
            
            # Salvar métricas
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            # Log das métricas de validação
            logger.info(f"Validação época {epoch + 1}:")
            logger.info(f"  - Loss: {val_loss:.4f}")
            logger.info(f"  - Acurácia: {val_accuracy:.4f}")
            logger.info(f"  - Precisão: {val_metrics['precision']:.4f}")
            logger.info(f"  - Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  - F1-Score: {val_metrics['f1']:.4f}")
            logger.info(f"  - AUC-ROC: {val_metrics['auc_roc']:.4f}")
            
            # Tempo da época
            epoch_duration = datetime.now() - epoch_start_time
            logger.info(f"Duração da época: {epoch_duration}")
            
            # Verificar overfitting
            if epoch > 0:
                train_loss_change = avg_train_loss - self.train_losses[epoch-1]
                val_loss_change = val_loss - self.val_losses[epoch-1]
                
                if train_loss_change < 0 and val_loss_change > 0:
                    logger.warning(f"Possível overfitting detectado - Loss treino diminuiu ({train_loss_change:.4f}) mas loss validação aumentou ({val_loss_change:.4f})")
                    
        logger.info("=== TREINAMENTO CONCLUÍDO ===")
    
    def evaluate(self, data_loader):
        """Avalia o modelo"""
        logger.info("Iniciando avaliação do modelo")
        eval_start_time = datetime.now()
        
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        logits_list = []
        
        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Avaliação"):
                batch_count += 1
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                logits_list.extend(torch.softmax(logits, dim=-1).cpu().numpy())
                
                # Log de progresso a cada 50 batches
                if batch_count % 50 == 0:
                    logger.debug(f"Avaliação: {batch_count}/{len(data_loader)} batches processados")
        
        logger.info(f"Processados {batch_count} batches de avaliação")
        
        # Calcular métricas
        logger.info("Calculando métricas de avaliação")
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # AUC-ROC
        probabilities = np.array(logits_list)[:, 1]  # Probabilidade da classe positiva
        auc_roc = roc_auc_score(true_labels, probabilities)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc
        }
        
        eval_duration = datetime.now() - eval_start_time
        logger.info(f"Avaliação concluída em {eval_duration}")
        logger.debug(f"Métricas calculadas: {metrics}")
        
        return avg_loss, accuracy, metrics
    
    def save_model(self, save_path, metrics, hyperparameters):
        """Salva o modelo e metadados"""
        logger.info(f"Iniciando salvamento do modelo em: {save_path}")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Diretório criado/verificado: {save_path}")
            
            # Salvar modelo e tokenizer
            logger.info("Salvando modelo BERT")
            self.model.save_pretrained(save_path)
            
            logger.info("Salvando tokenizer")
            self.tokenizer.save_pretrained(save_path)
            
            # Salvar métricas e hiperparâmetros
            logger.info("Salvando metadados")
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'max_length': self.max_length,
                'hyperparameters': hyperparameters,
                'metrics': metrics,
                'device': str(self.device),
                'training_history': {
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies
                }
            }
            
            metadata_path = os.path.join(save_path, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Modelo salvo com sucesso em: {save_path}")
            logger.info(f"Arquivos salvos:")
            for file in os.listdir(save_path):
                file_path = os.path.join(save_path, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                logger.info(f"  - {file}: {file_size:.2f} MB")
                
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            raise
    
    def load_model(self, load_path):
        """Carrega modelo salvo"""
        logger.info(f"Carregando modelo de: {load_path}")
        
        try:
            # Verificar se o caminho existe
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Caminho não encontrado: {load_path}")
            
            logger.info("Carregando modelo BERT")
            self.model = BertForSequenceClassification.from_pretrained(load_path).to(self.device)
            
            logger.info("Carregando tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained(load_path)
            
            logger.info("Carregando metadados")
            metadata_path = os.path.join(load_path, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Modelo carregado com sucesso:")
            logger.info(f"  - Modelo original: {metadata.get('model_name', 'N/A')}")
            logger.info(f"  - Timestamp: {metadata.get('timestamp', 'N/A')}")
            logger.info(f"  - F1-Score: {metadata.get('metrics', {}).get('f1', 'N/A'):.4f}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise



class ModelVersionManager:
    def __init__(self, base_path='models'):
        logger.info(f"Inicializando ModelVersionManager com base_path: {base_path}")
        
        self.base_path = base_path
        self.experiments_path = os.path.join(base_path, 'experiments')
        self.production_path = os.path.join(base_path, 'production')
        
        logger.info(f"Criando diretórios:")
        logger.info(f"  - Experimentos: {self.experiments_path}")
        logger.info(f"  - Produção: {self.production_path}")
        
        os.makedirs(self.experiments_path, exist_ok=True)
        os.makedirs(self.production_path, exist_ok=True)
        
        logger.info("ModelVersionManager inicializado com sucesso")
    
    def save_experiment(self, trainer, metrics, hyperparameters, experiment_name=None):
        """Salva experimento com timestamp"""
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Salvando experimento: {experiment_name}")
        experiment_path = os.path.join(self.experiments_path, experiment_name)
        
        logger.info(f"Caminho do experimento: {experiment_path}")
        logger.info(f"Métricas do experimento: F1={metrics.get('f1', 'N/A'):.4f}, Accuracy={metrics.get('accuracy', 'N/A'):.4f}")
        
        trainer.save_model(experiment_path, metrics, hyperparameters)
        
        logger.info(f"Experimento {experiment_name} salvo com sucesso")
        return experiment_path
    
    def promote_to_production(self, experiment_path, model_name='best_model'):
        """Promove modelo para produção se melhor que o atual"""
        logger.info(f"Avaliando promoção do modelo para produção")
        logger.info(f"Experimento: {experiment_path}")
        logger.info(f"Nome do modelo em produção: {model_name}")
        
        production_model_path = os.path.join(self.production_path, model_name)
        
        try:
            # Carregar métricas do experimento
            logger.info("Carregando métricas do experimento")
            with open(os.path.join(experiment_path, 'metadata.json'), 'r') as f:
                new_metadata = json.load(f)
            
            new_f1 = new_metadata['metrics']['f1']
            logger.info(f"F1-Score do novo modelo: {new_f1:.4f}")
            
            # Verificar se existe modelo em produção
            current_metadata_path = os.path.join(production_model_path, 'metadata.json')
            
            should_promote = True
            if os.path.exists(current_metadata_path):
                logger.info("Modelo em produção encontrado, comparando métricas")
                with open(current_metadata_path, 'r') as f:
                    current_metadata = json.load(f)
                
                # Comparar F1-score (ou outra métrica principal)
                current_f1 = current_metadata['metrics']['f1']
                logger.info(f"F1-Score do modelo atual em produção: {current_f1:.4f}")
                
                if new_f1 <= current_f1:
                    should_promote = False
                    logger.info(f"Novo modelo NÃO será promovido - F1 atual ({current_f1:.4f}) >= F1 novo ({new_f1:.4f})")
                else:
                    logger.info(f"Novo modelo SERA promovido - F1 novo ({new_f1:.4f}) > F1 atual ({current_f1:.4f})")
            else:
                logger.info("Nenhum modelo em produção encontrado - promovendo primeiro modelo")
            
            if should_promote:
                # Copiar modelo para produção
                import shutil
                logger.info(f"Copiando modelo para produção: {production_model_path}")
                
                if os.path.exists(production_model_path):
                    logger.info("Removendo modelo anterior em produção")
                    shutil.rmtree(production_model_path)
                    
                shutil.copytree(experiment_path, production_model_path)
                
                logger.info(f"SUCESSO: Modelo promovido para produção! F1-Score: {new_f1:.4f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erro durante promoção do modelo: {str(e)}")
            raise
    
    def list_experiments(self):
        """Lista todos os experimentos"""
        logger.info("Listando experimentos disponíveis")
        
        experiments = []
        try:
            exp_files = os.listdir(self.experiments_path)
            logger.info(f"Encontrados {len(exp_files)} itens no diretório de experimentos")
            
            for exp_name in exp_files:
                exp_path = os.path.join(self.experiments_path, exp_name)
                metadata_path = os.path.join(exp_path, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    experiment_info = {
                        'name': exp_name,
                        'timestamp': metadata['timestamp'],
                        'metrics': metadata['metrics']
                    }
                    experiments.append(experiment_info)
                    
                    logger.debug(f"Experimento: {exp_name} - F1: {metadata['metrics'].get('f1', 'N/A'):.4f}")
                else:
                    logger.warning(f"Metadata não encontrada para experimento: {exp_name}")
            
            logger.info(f"Total de experimentos válidos: {len(experiments)}")
            return experiments
            
        except Exception as e:
            logger.error(f"Erro ao listar experimentos: {str(e)}")
            return []


def main():
    """Função principal para execução do pipeline"""
    logger.info("=== PIPELINE DE FINE-TUNING BERT PARA ANÁLISE DE SENTIMENTOS ===")
    pipeline_start_time = datetime.now()
    logger.info(f"Início da execução: {pipeline_start_time}")
    
    # parametros para iniciar a classe do modelo BERT
    MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
    NUM_LABELS = 2
    MAX_LENGTH = 512

    # Parametros de treinamento
    BATCH_SIZE = 16  # Ajustado para 16 para evitar problemas de memória, apenas temos 8192 MiB de GPU
    EPOCHS = 1
    LEARNING_RATE = 2e-5

    # Load the dataset to be used for training and evaluation
    # This dataset is a collection of B2W reviews, which will be used for sentiment analysis
    # Note: Ensure that the dataset is available in the Hugging Face datasets library
    DATASET_HF = "ruanchaves/b2w-reviews01"
    
    # Hiperparâmetros
    hyperparameters = {
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE
        }
    
    # Log dos hiperparâmetros
    logger.info("=== CONFIGURAÇÕES DO PIPELINE ===")
    logger.info(f"Modelo: {MODEL_NAME}")
    logger.info(f"Dataset: {DATASET_HF}")
    logger.info(f"Hiperparâmetros: {hyperparameters}")
    
    # Inicializar trainer fazendo a carga do modelo BERT
    trainer = BERTSentimentTrainer(model_name=MODEL_NAME, max_length=MAX_LENGTH, num_labels=NUM_LABELS
                                   ,dataset_name=DATASET_HF)

    # Inicializar o gerenciador de versões
    version_manager = ModelVersionManager()
    
    # Carregar e preparar dados
    texts, labels = trainer.load_data()
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(texts, labels, seed_random_state=42)
    
    logger.info("=== RESUMO DOS DADOS PREPARADOS ===")
    logger.info(f"  Treino: {len(train_dataset)} amostras")
    logger.info(f"  Validação: {len(val_dataset)} amostras")
    logger.info(f"  Teste: {len(test_dataset)} amostras")
    logger.info(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} amostras")
    
    # Treinar modelo
    trainer.train(train_dataset, val_dataset, BATCH_SIZE, EPOCHS, LEARNING_RATE)
    
    # Avaliar no conjunto de teste
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    _, _, test_metrics = trainer.evaluate(test_loader)  # Only need test_metrics
    
    logger.info("\n=== RESULTADOS FINAIS NO CONJUNTO DE TESTE ===")
    logger.info(f"Loss de teste: {test_metrics['loss']:.4f}")
    logger.info(f"Acurácia: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precisão: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    # Log de resumo de performance
    if test_metrics['f1'] >= 0.8:
        logger.info("✅ EXCELENTE: F1-Score >= 0.8")
    elif test_metrics['f1'] >= 0.7:
        logger.info("✅ BOM: F1-Score >= 0.7")
    elif test_metrics['f1'] >= 0.6:
        logger.warning("⚠️ RAZOÁVEL: F1-Score >= 0.6")
    else:
        logger.warning("⚠️ BAIXO: F1-Score < 0.6 - Considere ajustar hiperparâmetros")
    
    # Salvar experimento
    experiment_path = version_manager.save_experiment(
        trainer, test_metrics, hyperparameters
    )
    
    # Tentar promover para produção
    version_manager.promote_to_production(experiment_path)
    
    # Calcular duração total do pipeline
    pipeline_duration = datetime.now() - pipeline_start_time
    
    logger.info(f"\n=== PIPELINE CONCLUÍDO ===")
    logger.info(f"Experimento salvo em: {experiment_path}")
    logger.info(f"Duração total do pipeline: {pipeline_duration}")
    logger.info(f"Timestamp de conclusão: {datetime.now()}")
    
    # Log final de sucesso
    logger.info("✅ Pipeline de fine-tuning BERT concluído com sucesso!")


if __name__ == "__main__":
    main()