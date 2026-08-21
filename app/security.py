import os
import json
import time
from base64 import b64encode, b64decode
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Request, HTTPException, Header

# Generate or load a 32-byte secret key
SECRET_KEY_HEX = os.environ.get("TOKEN_SECRET", os.urandom(32).hex())
SECRET_KEY = bytes.fromhex(SECRET_KEY_HEX)
aesgcm = AESGCM(SECRET_KEY)
MINIMUM_DELAY_MS = 1500  # 1.5 seconds

def generate_token(ip: str) -> str:
    nonce = os.urandom(12)
    payload = json.dumps({
        "ip": ip,
        "timestamp": int(time.time() * 1000)
    }).encode('utf-8')
    
    ciphertext = aesgcm.encrypt(nonce, payload, None)
    nonce_b64 = b64encode(nonce).decode('utf-8')
    ciphertext_b64 = b64encode(ciphertext).decode('utf-8')
    
    return f"{nonce_b64}:{ciphertext_b64}"

def verify_token(token: str) -> dict | None:
    try:
        nonce_b64, ciphertext_b64 = token.split(':')
        nonce = b64decode(nonce_b64)
        ciphertext = b64decode(ciphertext_b64)
        decrypted_payload = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(decrypted_payload.decode('utf-8'))
    except Exception:
        return None

# FastAPI Dependency for your chat route
async def verify_chat_token(request: Request, x_chat_token: str | None = Header(None)):
    client_ip = request.client.host
    if forwarded_for := request.headers.get("x-forwarded-for"):
        client_ip = forwarded_for.split(",")[0].strip()

    if not x_chat_token:
        raise HTTPException(status_code=403, detail="Missing security token.")

    decoded_payload = verify_token(x_chat_token)
    if not decoded_payload:
        raise HTTPException(status_code=403, detail="Invalid or tampered token.")

    if decoded_payload.get("ip") != client_ip:
        raise HTTPException(status_code=403, detail="IP mismatch.")

    time_since_last = int(time.time() * 1000) - decoded_payload.get("timestamp", 0)
    if time_since_last < MINIMUM_DELAY_MS:
        raise HTTPException(status_code=429, detail="Too many requests.")

    return client_ip