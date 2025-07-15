import requests
import json

BASE_URL = "http://localhost:8000"

def test_predict_positive():
    """Testa predição de sentimento positivo"""
    data = {"text": "Eu adorei o produto, a entrega foi muito rápida!"}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Teste positivo - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_predict_negative():
    """Testa predição de sentimento negativo"""
    data = {"text": "O produto é horrível, chegou quebrado e o atendimento é péssimo!"}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Teste negativo - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_empty_text():
    """Testa texto vazio - deve retornar erro 422"""
    data = {"text": ""}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Teste texto vazio - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 422

def test_whitespace_only_text():
    """Testa texto com apenas espaços em branco - deve retornar erro 400"""
    data = {"text": "   \t  \n  "}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Teste texto só espaços - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 400

def test_missing_text_field():
    """Testa campo text ausente - deve retornar erro 422"""
    data = {"message": "teste sem campo text"}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Teste campo ausente - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 422

def test_invalid_json():
    """Testa JSON inválido"""
    response = requests.post(
        f"{BASE_URL}/predict",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    print(f"Teste JSON inválido - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 422

def test_health_endpoint():
    """Testa endpoint de health check"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Teste health - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_root_endpoint():
    """Testa endpoint raiz"""
    response = requests.get(f"{BASE_URL}/")
    print(f"Teste root - Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

if __name__ == "__main__":
    print("=== Testando API de Análise de Sentimento ===\n")
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Sentimento Positivo", test_predict_positive),
        ("Sentimento Negativo", test_predict_negative),
        ("Texto Vazio", test_empty_text),
        ("Texto Só Espaços", test_whitespace_only_text),
        ("Campo Text Ausente", test_missing_text_field),
        ("JSON Inválido", test_invalid_json),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, "PASSOU" if result else "FALHOU"))
        except Exception as e:
            print(f"ERRO: {e}")
            results.append((test_name, "ERRO"))
        print()
    
    print("=== Resultados dos Testes ===")
    for test_name, result in results:
        print(f"{test_name}: {result}")