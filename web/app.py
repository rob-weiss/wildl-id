from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>wildl-id</title></head><body><main><h1>wildl-id</h1>"
        "</main></body></html>"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}