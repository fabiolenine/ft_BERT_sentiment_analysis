from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import logging
from typing import Dict, Any
import os
import time
import uuid
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/lenine/PycharmProjects/ft_BERT_sentiment_analysis/logs/api/api.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="API para análise de sentimento usando BERT fine-tuned",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Texto para análise de sentimento")

class PredictionResponse(BaseModel):
    sentiment: str = Field(..., description="Classificação do sentimento")
    score: float = Field(..., ge=0.0, le=1.0, description="Score de confiança da predição")

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_mapping = {0: "negativo", 1: "positivo"}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inicializando ModelLoader - Device: {self.device}")
        
    def load_model(self):
        start_time = time.time()
        try:
            logger.info(f"Iniciando carregamento do modelo - Path: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Caminho do modelo não encontrado: {self.model_path}")
                raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")
            
            logger.info("Carregando tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Tokenizer carregado com sucesso")
            
            logger.info("Carregando modelo BERT...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            logger.info(f"Movendo modelo para device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"Modelo carregado com sucesso em {load_time:.2f}s - Device: {self.device}")
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Falha ao carregar modelo após {load_time:.2f}s - Erro: {str(e)}", exc_info=True)
            raise RuntimeError(f"Falha ao carregar modelo: {e}")
    
    def predict(self, text: str, request_id: str = None) -> Dict[str, Any]:
        start_time = time.time()
        text_preview = text[:50] + "..." if len(text) > 50 else text
        
        if self.model is None or self.tokenizer is None:
            logger.error(f"Tentativa de predição com modelo não carregado - Request: {request_id}")
            raise RuntimeError("Modelo não foi carregado")
        
        try:
            logger.info(f"Iniciando predição - Request: {request_id} - Texto: '{text_preview}' - Tamanho: {len(text)} chars")
            
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.debug(f"Tokenização concluída - Request: {request_id} - Tokens: {inputs['input_ids'].shape}")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence_score = predictions[0][predicted_class].item()
            
            sentiment = self.label_mapping[predicted_class]
            prediction_time = time.time() - start_time
            
            logger.info(f"Predição concluída - Request: {request_id} - Resultado: {sentiment} - Score: {confidence_score:.4f} - Tempo: {prediction_time:.3f}s")
            
            return {
                "sentiment": sentiment,
                "score": round(confidence_score, 4)
            }
        
        except Exception as e:
            prediction_time = time.time() - start_time
            logger.error(f"Erro durante predição - Request: {request_id} - Tempo: {prediction_time:.3f}s - Erro: {str(e)}", exc_info=True)
            raise RuntimeError(f"Erro durante predição: {e}")

model_path = "/home/lenine/PycharmProjects/ft_BERT_sentiment_analysis/models/production/best_model"
model_loader = ModelLoader(model_path)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"Incoming request - ID: {request_id} - Method: {request.method} - URL: {request.url} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Request completed - ID: {request_id} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

@app.on_event("startup")
async def startup_event():
    start_time = time.time()
    try:
        logger.info("=== Iniciando API BERT Sentiment Analysis ===")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Python version: {torch.__version__}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA disponível: {torch.cuda.is_available()}")
        
        model_loader.load_model()
        
        startup_time = time.time() - start_time
        logger.info(f"=== API iniciada com sucesso em {startup_time:.2f}s ===")
        
    except Exception as e:
        startup_time = time.time() - start_time
        logger.error(f"=== Falha crítica ao inicializar API após {startup_time:.2f}s ===", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=== Encerrando API BERT Sentiment Analysis ===")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest, http_request: Request):
    request_id = str(uuid.uuid4())[:8]
    client_ip = http_request.client.host
    
    try:
        logger.info(f"Nova solicitação de predição - Request: {request_id} - Client: {client_ip}")
        
        if not request.text.strip():
            logger.warning(f"Texto vazio rejeitado - Request: {request_id} - Client: {client_ip}")
            raise HTTPException(
                status_code=400,
                detail="Campo 'text' não pode estar vazio"
            )
        
        result = model_loader.predict(request.text, request_id)
        
        response = PredictionResponse(
            sentiment=result["sentiment"],
            score=result["score"]
        )
        
        logger.info(f"Predição entregue com sucesso - Request: {request_id} - Client: {client_ip}")
        return response
    
    except HTTPException as e:
        logger.warning(f"Erro de validação - Request: {request_id} - Client: {client_ip} - Status: {e.status_code} - Detail: {e.detail}")
        raise
    except ValueError as e:
        logger.error(f"Erro de dados inválidos - Request: {request_id} - Client: {client_ip} - Erro: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Dados de entrada inválidos: {str(e)}"
        )
    except RuntimeError as e:
        logger.error(f"Erro de runtime - Request: {request_id} - Client: {client_ip} - Erro: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno do servidor: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erro não esperado - Request: {request_id} - Client: {client_ip} - Erro: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Erro interno do servidor"
        )

@app.get("/health")
async def health_check(request: Request):
    client_ip = request.client.host
    logger.debug(f"Health check solicitado - Client: {client_ip}")
    
    health_status = {
        "status": "healthy" if model_loader.model is not None else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loader.model is not None,
        "device": str(model_loader.device),
        "model_path": model_loader.model_path
    }
    
    if model_loader.model is None:
        logger.warning(f"Health check: modelo não carregado - Client: {client_ip}")
    
    return health_status

@app.get("/")
async def root(request: Request):
    client_ip = request.client.host
    logger.debug(f"Root endpoint acessado - Client: {client_ip}")
    
    return {
        "message": "API de Análise de Sentimento com BERT",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "status": "online",
        "endpoints": {
            "/predict": "POST - Análise de sentimento",
            "/health": "GET - Status da API",
            "/docs": "GET - Documentação Swagger"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)