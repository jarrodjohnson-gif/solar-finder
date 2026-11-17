"""Solar Finder web app: a browser tool at /tool and a JSON API for other tools.

Run (from the repo root):
    uvicorn --app-dir src solar_detector.api.app:app --host 127.0.0.1 --port 8002

Endpoints
    GET  /tool               page: type an address or upload a roof image
    POST /api/detect         JSON {address} or {lat, lng}; optional zoom, conf
    POST /api/detect-image   multipart upload (field "image"); optional conf
    GET  /health             model + key status

Address lookups need GOOGLE_MAPS_API_KEY (satellite imagery) in the repo's .env.
Image uploads work without any key. The web app never downloads model weights by itself:
it uses the trained model if it is already installed (`solar-finder-3000 --clear-cache` then a
CLI run downloads it), otherwise the built-in colour/shape detector.
"""
from __future__ import annotations

import base64
import io
import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

from .. import detect_solar_panels, fetch_satellite_image, geocode, GeocodingError, ImageryError
from ..model_manager import is_solar_model_available

app = FastAPI(title="Solar Finder", version="1.0")


def _key(name: str) -> str | None:
    v = (os.getenv(name) or "").strip()
    return None if (not v or v.startswith("your_")) else v


def _result(res, **extra) -> dict:
    img_b64 = None
    if res.annotated_image is not None:
        buf = io.BytesIO()
        res.annotated_image.save(buf, format="JPEG", quality=88)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "has_solar_panels": res.has_solar_panels,
        "confidence": round(float(res.confidence), 4),
        "num_panels": int(res.num_panels),
        "method": res.method,            # "yolo" (trained model) or "colour" (fallback)
        "annotated_image_jpeg_base64": img_b64,
        **extra,
    }


class DetectRequest(BaseModel):
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    zoom: int = Field(20, ge=17, le=21)
    conf: float = Field(0.3, ge=0.05, le=0.95)


@app.get("/")
def root():
    return RedirectResponse("/tool")


@app.get("/health")
def health():
    return {
        "ok": True,
        "solar_model_cached": is_solar_model_available(),
        "maps_key_set": bool(_key("GOOGLE_MAPS_API_KEY")),
        "geocoding": "google" if _key("GOOGLE_GEOCODING_API_KEY") else "nominatim (free)",
    }


@app.post("/api/detect")
def detect(req: DetectRequest):
    if not _key("GOOGLE_MAPS_API_KEY"):
        raise HTTPException(503, "GOOGLE_MAPS_API_KEY is not set in solar-finder/.env, so satellite images "
                                 "can't be fetched. Upload an image instead, or add a new key.")
    t0 = time.time()
    if req.lat is not None and req.lng is not None:
        lat, lng = req.lat, req.lng
    elif req.address and req.address.strip():
        try:
            lat, lng = geocode(req.address.strip())
        except GeocodingError as e:
            raise HTTPException(422, f"Couldn't find that address: {e}")
    else:
        raise HTTPException(422, "Give an address, or lat and lng.")
    try:
        image = fetch_satellite_image(lat, lng, zoom=req.zoom)
    except ImageryError as e:
        raise HTTPException(502, f"Satellite image failed: {e}")
    res = detect_solar_panels(image, conf_threshold=req.conf, annotate=True)
    return _result(res, lat=lat, lng=lng, address=req.address, seconds=round(time.time() - t0, 2))


@app.post("/api/detect-image")
async def detect_image(image: UploadFile = File(...), conf: float = Form(0.3)):
    data = await image.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image too large (15 MB max).")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(422, "That file isn't an image.")
    t0 = time.time()
    res = detect_solar_panels(img, conf_threshold=max(0.05, min(conf, 0.95)), annotate=True)
    return _result(res, filename=image.filename, seconds=round(time.time() - t0, 2))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Solar Finder</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2433;--mute:#667085;--line:#e4e7ec;--acc:#5e5ce6;--yes:#12b76a;--no:#98a2b3;--warn:#b54708}
@media (prefers-color-scheme:dark){:root{--bg:#101318;--card:#181c23;--ink:#e6e9ef;--mute:#98a2b3;--line:#2a303a;--acc:#8b8af7;--warn:#fdb022}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.row{display:flex;gap:8px;flex-wrap:wrap}input[type=text]{flex:1;min-width:220px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:transparent;color:inherit;font:inherit}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}button:disabled{opacity:.5;cursor:wait}
label.up{display:inline-block;padding:10px 14px;border:1px dashed var(--line);border-radius:8px;cursor:pointer;color:var(--mute)}input[type=file]{display:none}
.warn{color:var(--warn);font-size:13px;margin-top:8px}.verdict{font-size:20px;font-weight:700}.yes{color:var(--yes)}.no{color:var(--no)}
.meta{color:var(--mute);font-size:13px;margin-top:4px}img.out{width:100%;border-radius:8px;margin-top:12px;border:1px solid var(--line)}.err{color:#d92d20}
</style></head><body><main>
<h1>Solar Finder</h1><p class="sub">Checks a roof for solar panels from a satellite image.</p>
<div class="card"><div class="row"><input id="addr" type="text" placeholder="Property address, e.g. 123 Main St, Austin TX" autofocus>
<button id="go">Check</button></div><div id="keywarn" class="warn" hidden>Address lookup needs a Google Maps key in solar-finder/.env. Uploading a roof image works now.</div></div>
<div class="card"><label class="up">Or upload a roof / satellite image<input id="file" type="file" accept="image/*"></label></div>
<div id="out" class="card" hidden></div>
<script>
const $=id=>document.getElementById(id);
fetch('/health').then(r=>r.json()).then(h=>{if(!h.maps_key_set){$('keywarn').hidden=false}});
function show(d){const o=$('out');o.hidden=false;
 if(d.detail){o.innerHTML='<div class="err">'+d.detail+'</div>';return}
 const v=d.has_solar_panels?'<span class="verdict yes">Solar panels detected</span>':'<span class="verdict no">No solar panels detected</span>';
 const m=(d.has_solar_panels?(d.confidence*100).toFixed(1)+'% confidence, '+d.num_panels+' panel region(s). ':'')+'Method: '+(d.method==='yolo'?'trained model':'colour fallback')+' · '+d.seconds+' s';
 o.innerHTML=v+'<div class="meta">'+m+'</div>'+(d.annotated_image_jpeg_base64?'<img class="out" src="data:image/jpeg;base64,'+d.annotated_image_jpeg_base64+'">':'')}
async function run(p){$('go').disabled=true;$('out').hidden=false;$('out').innerHTML='Checking…';
 try{const r=await p;show(await r.json())}catch(e){show({detail:String(e)})}finally{$('go').disabled=false}}
$('go').onclick=()=>{const a=$('addr').value.trim();if(!a)return;run(fetch('/api/detect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({address:a})}))};
$('addr').onkeydown=e=>{if(e.key==='Enter')$('go').click()};
$('file').onchange=()=>{const f=$('file').files[0];if(!f)return;const fd=new FormData();fd.append('image',f);run(fetch('/api/detect-image',{method:'POST',body:fd}))};
const q=new URLSearchParams(location.search).get('address');if(q){$('addr').value=q;$('go').click()}
</script></main></body></html>"""


@app.get("/tool", response_class=HTMLResponse)
def tool():
    return PAGE
