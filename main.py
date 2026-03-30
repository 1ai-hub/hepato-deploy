from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model_utils import predict_toxicity

app = FastAPI(title="Axleres AI Hepatotoxicity Predictor", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Axleres AI | Hepatotoxicity Predictor</title>
        <style>
            :root {
                --purple: #7c3aed;
                --violet: #9333ea;
                --gold: #d9a923;
                --bg1: #fcfbff;
                --bg2: #fffaf2;
                --bg3: #fdf5ff;
                --text: #1f2937;
                --muted: #6b7280;
                --border: rgba(124, 58, 237, 0.10);
                --card: rgba(255, 255, 255, 0.84);
            }

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at 12% 20%, rgba(217, 169, 35, 0.10), transparent 24%),
                    radial-gradient(circle at 88% 18%, rgba(147, 51, 234, 0.10), transparent 22%),
                    radial-gradient(circle at 70% 82%, rgba(217, 169, 35, 0.08), transparent 20%),
                    linear-gradient(135deg, var(--bg1), var(--bg2), var(--bg3));
                padding: 18px 18px 32px 18px;
            }

            .shell {
                width: 100%;
                max-width: 1120px;
                margin: 0 auto;
                position: relative;
            }

            .navbar {
            position: fixed;   /* NOT absolute */
            top: 16px;
            left: 20px;
            z-index: 1000;
            }
            .nav-left img {
            height: 110px;   /* bigger */
            width: auto;
            object-fit: contain;
            display: block;
            }

            .hero {
                text-align: center;
                padding-top: 82px;
                margin-bottom: 22px;
            }

            .hero h1 {
                margin: 0 0 10px 0;
                font-size: 2.15rem;
                font-weight: 760;
                letter-spacing: -0.03em;
            }

            .hero p {
                margin: 0 auto;
                max-width: 760px;
                color: var(--muted);
                line-height: 1.65;
                font-size: 1.02rem;
            }

            .grid {
                display: grid;
                grid-template-columns: 1.05fr 0.95fr;
                gap: 22px;
                align-items: start;
            }

            .card {
                background: var(--card);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow:
                    0 10px 30px rgba(17, 24, 39, 0.06),
                    0 3px 10px rgba(17, 24, 39, 0.04);
                padding: 28px;
            }

            .card h2 {
                margin: 0 0 18px 0;
                font-size: 1.15rem;
                font-weight: 700;
            }

            label {
                display: block;
                font-size: 0.94rem;
                font-weight: 650;
                margin-bottom: 8px;
            }

            .hint {
                color: var(--muted);
                font-size: 0.87rem;
                margin-top: -2px;
                margin-bottom: 14px;
                line-height: 1.5;
            }

            input[type="text"],
            input[type="file"] {
                width: 100%;
                border: 1px solid rgba(107, 114, 128, 0.18);
                background: rgba(255, 255, 255, 0.92);
                border-radius: 16px;
                padding: 13px 15px;
                font-size: 0.96rem;
                outline: none;
                transition: 0.2s ease;
                margin-bottom: 16px;
            }

            input[type="text"]:focus,
            input[type="file"]:focus {
                border-color: rgba(124, 58, 237, 0.40);
                box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.10);
            }

            .divider {
                display: flex;
                align-items: center;
                gap: 12px;
                color: var(--muted);
                font-size: 0.84rem;
                margin: 6px 0 16px 0;
            }

            .divider::before,
            .divider::after {
                content: "";
                height: 1px;
                background: rgba(107, 114, 128, 0.18);
                flex: 1;
            }

            button {
                width: 100%;
                border: none;
                border-radius: 16px;
                padding: 14px 16px;
                font-size: 1rem;
                font-weight: 750;
                color: white;
                cursor: pointer;
                background: linear-gradient(90deg, var(--violet), #b453c6 45%, var(--gold));
                box-shadow: 0 10px 22px rgba(124, 58, 237, 0.16);
                transition: transform 0.15s ease, opacity 0.15s ease;
                margin-top: 6px;
            }

            button:hover {
                opacity: 0.97;
                transform: translateY(-1px);
            }

            button:disabled {
                opacity: 0.7;
                cursor: wait;
                transform: none;
            }

            .pill-row {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-top: 18px;
            }

            .pill {
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 650;
                background: rgba(124, 58, 237, 0.06);
                color: var(--purple);
                border: 1px solid rgba(124, 58, 237, 0.12);
            }

            .result-box {
                background: rgba(248, 250, 252, 0.95);
                border: 1px solid rgba(107, 114, 128, 0.10);
                border-radius: 18px;
                padding: 18px;
                min-height: 290px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-word;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.92rem;
                line-height: 1.55;
            }

            .footer-note {
                margin-top: 14px;
                font-size: 0.84rem;
                color: var(--muted);
                line-height: 1.55;
            }

            @media (max-width: 860px) {
                body {
                    padding: 16px 14px 24px 14px;
                }

                .navbar {
                    position: static;
                    margin-bottom: 6px;
                }

                .nav-left img {
                    height: 72px;
                }

                .hero {
                    padding-top: 6px;
                    margin-bottom: 18px;
                }

                .hero h1 {
                    font-size: 1.8rem;
                }

                .grid {
                    grid-template-columns: 1fr;
                }

                .card {
                    padding: 22px;
                }
            }
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="navbar">
                <div class="nav-left">
                    <img src="/static/logo.png" alt="Axleres AI logo" />
                </div>
            </div>

            <div class="hero">
                <h1>Hepatotoxicity Predictor</h1>
                <p>
                    Predict liver toxicity risk directly from molecular structure using the
                    Axleres AI chemistry-based screening pipeline.
                </p>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>Run Prediction</h2>

                    <form id="predictForm">
                        <label for="smiles">SMILES input</label>
                        <div class="hint">
                            Paste a valid SMILES string for single-molecule prediction.
                        </div>
                        <input
                            type="text"
                            id="smiles"
                            name="smiles"
                            placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
                        />

                        <div class="divider">or</div>

                        <label for="file">Upload structure file</label>
                        <div class="hint">Accepted formats: .mol or .sdf</div>
                        <input type="file" id="file" name="file" accept=".mol,.sdf" />

                        <button id="submitBtn" type="submit">Run Prediction</button>
                    </form>
                </div>

                <div class="card">
                    <h2>Result</h2>
                    <div id="result" class="result-box">Waiting for input...</div>
                    <div class="footer-note">
                        Output includes toxicity probability, toxicity percentage, and risk category.
                    </div>
                </div>
            </div>
        </div>

        <script>
            const form = document.getElementById("predictForm");
            const result = document.getElementById("result");
            const submitBtn = document.getElementById("submitBtn");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();

                result.textContent = "Running prediction...";
                submitBtn.disabled = true;
                submitBtn.textContent = "Processing...";

                const formData = new FormData();
                const smiles = document.getElementById("smiles").value.trim();
                const fileInput = document.getElementById("file");

                if (smiles) {
                    formData.append("smiles", smiles);
                }

                if (fileInput.files.length > 0) {
                    formData.append("file", fileInput.files[0]);
                }

                if (!smiles && fileInput.files.length === 0) {
                    result.textContent = "Please enter a SMILES string or upload a MOL/SDF file.";
                    submitBtn.disabled = false;
                    submitBtn.textContent = "Run Prediction";
                    return;
                }

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    result.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    result.textContent = "Error: " + err.message;
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = "Run Prediction";
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(
    smiles: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    try:
        if smiles:
            result = predict_toxicity(smiles=smiles)
        elif file:
            content = await file.read()
            result = predict_toxicity(file_bytes=content, filename=file.filename)
        else:
            raise HTTPException(status_code=400, detail="Provide either a SMILES string or a MOL/SDF file.")

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")