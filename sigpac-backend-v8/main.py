from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIGPAC Sentinel API", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SIGPAC_CONSULTA_URL  = "https://sigpac-hubcloud.es/servicioconsultassigpac/query"
COPERNICUS_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Sentinel Hub Process API (requiere token OAuth)
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
# WMS de Copernicus Data Space
WMS_URL = "https://sh.dataspace.copernicus.eu/ogc/wms"

COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")
_token_cache = {"token": None, "expires_at": 0}


def cache_key(prefix: str, **kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()


async def get_copernicus_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    if not COPERNICUS_USER or not COPERNICUS_PASS:
        raise HTTPException(status_code=500, detail="Credenciales Copernicus no configuradas.")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COPERNICUS_TOKEN_URL,
            data={"grant_type": "password", "username": COPERNICUS_USER,
                  "password": COPERNICUS_PASS, "client_id": "cdse-public"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        logger.info("Token Copernicus obtenido")
        return _token_cache["token"]


def bbox_to_float(bbox_str: str):
    return list(map(float, bbox_str.split(",")))


async def procesar_sentinel_evalscript(
    bbox: str,
    fecha: str,
    evalscript: str,
    token: str,
    width: int = 2560,
    height: int = 2560,
) -> Optional[bytes]:
    """
    Usa la Sentinel Hub Process API para obtener imágenes procesadas.
    Esta API SÍ funciona con el token OAuth de Copernicus Data Space.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_to_float(bbox)

    # Fecha inicio y fin (día completo)
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    fecha_inicio = fecha_dt.strftime("%Y-%m-%dT00:00:00Z")
    fecha_fin = (fecha_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    payload = {
        "input": {
            "bounds": {
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": fecha_inicio, "to": fecha_fin},
                    "maxCloudCoverage": 80,
                }
            }]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
        },
        "evalscript": evalscript,
    }

    try:
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/png",
            },
            timeout=120,
        ) as client:
            resp = await client.post(PROCESS_API_URL, json=payload)
            logger.info(f"Process API status: {resp.status_code}")
            if resp.status_code == 200:
                return resp.content
            else:
                logger.error(f"Process API error: {resp.text[:300]}")
    except Exception as e:
        logger.error(f"Error Process API: {e}")
    return None


# Evalscripts para cada índice
EVALSCRIPTS = {
    "RGB": """
//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  // Corrección gamma + contraste mejorado
  // 1. Ganancia adaptativa
  let r = sample.B04 * 3.0;
  let g = sample.B03 * 3.0;
  let b = sample.B02 * 3.5;  // Azul ligeramente más amplificado
  // 2. Clip a [0,1]
  r = Math.min(Math.max(r, 0), 1);
  g = Math.min(Math.max(g, 0), 1);
  b = Math.min(Math.max(b, 0), 1);
  // 3. Corrección gamma 0.75 (aclara sombras sin quemar luces)
  let gamma = 0.75;
  return [Math.pow(r, gamma), Math.pow(g, gamma), Math.pow(b, gamma)];
}
""",
    "NDVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-10);
  if (ndvi < -0.5) return [0.05, 0.05, 0.05];
  else if (ndvi < 0) return [0.75, 0.75, 0.75];
  else if (ndvi < 0.1) return [0.86, 0.86, 0.86];
  else if (ndvi < 0.2) return [1, 1, 0.88];
  else if (ndvi < 0.3) return [0.86, 0.96, 0.72];
  else if (ndvi < 0.4) return [0.56, 0.82, 0.54];
  else if (ndvi < 0.5) return [0.27, 0.67, 0.36];
  else if (ndvi < 0.6) return [0.13, 0.52, 0.26];
  else if (ndvi < 0.7) return [0.05, 0.39, 0.16];
  else return [0.0, 0.27, 0.09];
}
""",
    "NDWI": """
//VERSION=3
function setup() {
  return { input: ["B03", "B08"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 1e-10);
  let val = (ndwi + 1) / 2;
  return [1 - val, 1 - val, val];
}
""",
    "EVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04", "B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let evi = 2.5 * (sample.B08 - sample.B04) / (sample.B08 + 6*sample.B04 - 7.5*sample.B02 + 1 + 1e-10);
  let val = Math.min(Math.max((evi + 1) / 2, 0), 1);
  return [1 - val, val, 1 - val];
}
""",
    "NDRE": """
//VERSION=3
function setup() {
  return { input: ["B08", "B05"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-10);
  let val = Math.min(Math.max((ndre + 1) / 2, 0), 1);
  return [1 - val, val, 0.5 - val * 0.5];
}
""",
    "SAVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let savi = 1.5 * (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.5 + 1e-10);
  let val = Math.min(Math.max((savi + 1) / 2, 0), 1);
  return [1 - val, val, 0.2];
}
""",
}

INDICES_INFO = {
    "NDVI": "Normalized Difference Vegetation Index",
    "NDWI": "Normalized Difference Water Index",
    "EVI":  "Enhanced Vegetation Index",
    "NDRE": "Normalized Difference Red Edge",
    "SAVI": "Soil-Adjusted Vegetation Index",
}


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "8.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "copernicus_configured": bool(COPERNICUS_USER and COPERNICUS_PASS),
    }


@app.get("/sigpac/punto")
async def get_parcela_por_punto(lat: float = Query(...), lon: float = Query(...)):
    ck = cache_key("sigpac_punto", lat=round(lat, 6), lon=round(lon, 6))
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"
    if cache_file.exists():
        return JSONResponse(content=json.loads(cache_file.read_text()))

    url = f"{SIGPAC_CONSULTA_URL}/recinfobypoint/4326/{lon}/{lat}.geojson"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        if not data.get("features"):
            raise HTTPException(status_code=404, detail="No se encontró parcela.")
        cache_file.write_text(json.dumps(data))
        return JSONResponse(content=data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error SIGPAC: {str(e)}")


@app.get("/sentinel/buscar")
async def buscar_imagenes(
    bbox: str = Query(...),
    fecha_inicio: str = Query(...),
    fecha_fin: str = Query(...),
    max_nubosidad: float = Query(30.0),
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox invalido")

    footprint = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and ContentDate/Start gt {fecha_inicio}T00:00:00.000Z "
            f"and ContentDate/Start lt {fecha_fin}T23:59:59.000Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_nubosidad}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
        "$expand": "Attributes",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(COPERNICUS_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        productos = []
        for item in data.get("value", []):
            cloud = next((a["Value"] for a in item.get("Attributes", []) if a["Name"] == "cloudCover"), None)
            productos.append({
                "id": item["Id"],
                "nombre": item["Name"],
                "fecha": item["ContentDate"]["Start"][:10],
                "nubosidad": round(cloud, 1) if cloud is not None else None,
                "size_mb": round(item.get("ContentLength", 0) / 1e6, 1),
            })
        return {"total": len(productos), "productos": productos}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error Copernicus: {e}")


@app.get("/imagen/rgb")
async def imagen_rgb(
    bbox: str = Query(...),
    fecha: str = Query(..., description="YYYY-MM-DD"),
):
    """Imagen en color natural usando Sentinel Hub Process API."""
    ck = cache_key("rgb6", bbox=bbox, fecha=fecha)
    cache_png = CACHE_DIR / f"{ck}_rgb.png"

    if cache_png.exists():
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_rgb(cache_png)

    png_bytes = await procesar_sentinel_evalscript(bbox, fecha, EVALSCRIPTS["RGB"], token)

    if not png_bytes:
        logger.warning("Process API falló para RGB, usando demo")
        return _demo_rgb(cache_png)

    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/indices/lista")
async def lista_indices():
    return {k: {"descripcion": v, "evalscript": True} for k, v in INDICES_INFO.items()}


@app.get("/indice/calcular")
async def calcular_indice(
    bbox: str = Query(...),
    fecha: str = Query(..., description="YYYY-MM-DD"),
    indice: str = Query(...),
    formato: str = Query("png"),
):
    """Calcula índice usando Sentinel Hub Process API con evalscript."""
    indice = indice.upper()
    if indice not in EVALSCRIPTS:
        raise HTTPException(status_code=400, detail=f"Indice desconocido: {list(INDICES_INFO.keys())}")

    ck = cache_key("indice6", bbox=bbox, fecha=fecha, idx=indice)
    cache_png = CACHE_DIR / f"{ck}.png"
    cache_stats = CACHE_DIR / f"{ck}_stats.json"

    if cache_png.exists() and formato == "png":
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")
    if cache_stats.exists() and formato == "stats":
        return JSONResponse(content=json.loads(cache_stats.read_text()))

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_indice_simple(indice, cache_png, cache_stats, formato)

    png_bytes = await procesar_sentinel_evalscript(bbox, fecha, EVALSCRIPTS[indice], token)

    if not png_bytes:
        return _demo_indice_simple(indice, cache_png, cache_stats, formato)

    # Calcula estadísticas aproximadas desde la imagen
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    verde = arr[:, :, 1]  # Canal verde como proxy del índice
    stats = {
        "indice": indice,
        "min": float(np.nanmin(verde)),
        "max": float(np.nanmax(verde)),
        "mean": float(np.nanmean(verde)),
        "std": float(np.nanstd(verde)),
    }
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


def _demo_rgb(cache_png: Path):
    np.random.seed(123)
    size = (256, 256)
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    base = 0.5 + 0.3 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2)
    r = np.clip(base * 80 + np.random.normal(0, 5, size), 50, 130).astype(np.uint8)
    g = np.clip(base * 120 + np.random.normal(0, 5, size), 80, 180).astype(np.uint8)
    b = np.clip(base * 50 + np.random.normal(0, 5, size), 30, 90).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


def _demo_indice_simple(indice, cache_png, cache_stats, formato):
    np.random.seed(42)
    size = (256, 256)
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    base = 0.3 + 0.4 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.15)
    vals = np.clip(base + np.random.normal(0, 0.05, size), 0, 1).astype(np.float32)

    stats = {"indice": indice, "min": float(vals.min()), "max": float(vals.max()),
             "mean": float(vals.mean()), "std": float(vals.std()), "modo": "DEMO"}
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    cmaps = {"NDVI": "RdYlGn", "NDWI": "Blues", "EVI": "YlGn", "NDRE": "RdYlGn", "SAVI": "YlGn"}
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('#0a0f0d')
    ax.set_facecolor('#0a1a0d')
    im = ax.imshow(vals, cmap=cmaps.get(indice, "RdYlGn"), vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{indice} - DEMO", color='#e2ffe8', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0f0d')
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/cache/info")
async def cache_info():
    files = list(CACHE_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return {"archivos": len(files), "total_mb": round(total_mb, 2)}


@app.delete("/cache/limpiar")
async def limpiar_cache(dias: int = Query(7)):
    cutoff = time.time() - dias * 86400
    eliminados = sum(1 for f in CACHE_DIR.glob("*") if f.is_file() and f.stat().st_mtime < cutoff and not f.unlink())
    return {"eliminados": eliminados}
