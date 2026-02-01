import json
import os
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import asyncio

# Add project root to path so we can import benchmark, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
GAMES_DIR = PROJECT_ROOT / "games"

# In-memory store for running benchmarks
_runs: dict[str, list[dict]] = {}


@app.get("/api/benchmarks")
def list_benchmarks():
    if not BENCHMARKS_DIR.exists():
        return []
    files = sorted(BENCHMARKS_DIR.glob("benchmark_*.json"), reverse=True)
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            result.append({
                "filename": f.name,
                "timestamp": data.get("timestamp", ""),
                "num_games": data.get("num_games", 0),
                "summary": data.get("summary", {}),
            })
        except Exception:
            continue
    return result


@app.get("/api/benchmarks/{filename}")
def get_benchmark(filename: str):
    path = BENCHMARKS_DIR / filename
    if not path.exists() or not path.name.startswith("benchmark_"):
        return {"error": "not found"}
    return json.loads(path.read_text())


class BenchmarkRequest(BaseModel):
    num_games: int = 5
    workers: int = 5


@app.post("/api/benchmarks/run")
def run_benchmark_endpoint(req: BenchmarkRequest):
    run_id = str(uuid.uuid4())
    _runs[run_id] = []

    def run():
        try:
            from benchmark import _run_single_game
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results_list = [None] * req.num_games
            with ThreadPoolExecutor(max_workers=req.workers) as pool:
                futures = {
                    pool.submit(_run_single_game, i + 1): i
                    for i in range(req.num_games)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    game = future.result()
                    results_list[idx] = game
                    _runs[run_id].append({
                        "game": idx + 1,
                        "result": game["result"],
                        "reason": game["reason"],
                        "moves": game["moves"],
                    })

            # Save using the benchmark module's format
            from benchmark import run_benchmark as _  # noqa: just to confirm importable
            from datetime import datetime
            total_hits = sum(g["hits"] for g in results_list if g)
            total_misses = sum(g["misses"] for g in results_list if g)
            total = total_hits + total_misses
            n = req.num_games

            report = {
                "timestamp": datetime.now().isoformat(),
                "mode": "retrieval + random fallback",
                "num_games": n,
                "workers": req.workers,
                "summary": {
                    "sunfish_wins": sum(1 for g in results_list if g and g["result"] == "1-0"),
                    "retrieval_wins": sum(1 for g in results_list if g and g["result"] == "0-1"),
                    "draws": sum(1 for g in results_list if g and g["result"] == "1/2-1/2"),
                    "avg_moves": round(sum(g["moves"] for g in results_list if g) / n),
                    "avg_sunfish_ms": round(sum(g["avg_sunfish_ms"] for g in results_list if g) / n),
                    "avg_retrieval_ms": round(sum(g["avg_retrieval_ms"] for g in results_list if g) / n),
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "hit_rate_pct": round((total_hits / total * 100) if total else 0, 1),
                },
                "games": results_list,
            }

            BENCHMARKS_DIR.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = BENCHMARKS_DIR / f"benchmark_{ts}.json"
            path.write_text(json.dumps(report, indent=2))

            _runs[run_id].append({"done": True, "filename": path.name})
        except Exception as e:
            _runs[run_id].append({"done": True, "error": str(e)})

    threading.Thread(target=run, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/benchmarks/stream/{run_id}")
async def stream_benchmark(run_id: str):
    async def generate():
        seen = 0
        while True:
            events = _runs.get(run_id, [])
            while seen < len(events):
                yield {"data": json.dumps(events[seen])}
                if events[seen].get("done"):
                    return
                seen += 1
            await asyncio.sleep(0.5)

    return EventSourceResponse(generate())


@app.get("/api/games")
def list_games():
    if not GAMES_DIR.exists():
        return []
    files = sorted(GAMES_DIR.glob("*.pgn"), reverse=True)
    return [f.name for f in files]


@app.get("/api/games/{filename}")
def get_game(filename: str):
    path = GAMES_DIR / filename
    if not path.exists() or not path.suffix == ".pgn":
        return {"error": "not found"}
    return {"pgn": path.read_text(), "filename": filename}
