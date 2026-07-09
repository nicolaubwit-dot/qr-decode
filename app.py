import io
import os
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode
from fastapi import FastAPI, UploadFile, File, Header, HTTPException

app = FastAPI(title="QR decode service")

API_KEY = os.environ.get("QR_API_KEY")

def _try(gray):
    for scale in (1.0, 0.75, 0.6, 0.5, 0.4, 0.3):
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for k in (0, 3, 5, 7):
            gb = cv2.GaussianBlur(g, (k, k), 0) if k else g
            for method in ("otsu", "raw", "adapt"):
                if method == "otsu":
                    _, im = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif method == "adapt":
                    im = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
                else:
                    im = gb
                res = zbar_decode(Image.fromarray(im))
                if res:
                    return res[0].data.decode("utf-8", "replace")
    return None

def decode_bytes(raw):
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = _try(gray)
    if data is None:
        try:
            det = cv2.QRCodeDetector()
            d, _, _ = det.detectAndDecode(img)
            if d:
                data = d
        except Exception:
            pass
    return data

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

# Same path and same response shape as goQR, so in Make you only swap the URL.
@app.post("/v1/read-qr-code/")
async def read_qr(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    data = decode_bytes(await file.read())
    err = None if data else "could not find/read QR Code"
    return [{"type": "qrcode", "symbol": [{"seq": 0, "data": data, "error": err}]}]
