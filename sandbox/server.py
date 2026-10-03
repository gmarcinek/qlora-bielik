"""HTTP API of the sandbox sidecar. Only the lab API reaches it (internal Docker network + bearer token)."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from sandbox.preview import PreviewError
from sandbox.preview import preview as build_preview
from sandbox.workspace import SandboxError, Workspace

TOKEN = os.getenv("SANDBOX_TOKEN", "")
MAX_UPLOAD_BYTES = int(os.getenv("SANDBOX_MAX_UPLOAD_MB", "300")) * 1024 * 1024
workspace = Workspace(Path(os.getenv("SANDBOX_ROOT", "/sessions")), Path(os.getenv("SHARED_ROOT", "/shared")))


def authorize(authorization: str = Header(default="")) -> None:
    if not TOKEN or not hmac.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(status_code=401, detail="Brak autoryzacji.")


app = FastAPI(title="Bielik sandbox", dependencies=[Depends(authorize)])


@app.exception_handler(SandboxError)
def sandbox_error(_: Request, error: SandboxError) -> JSONResponse:
    return JSONResponse(status_code=error.status, content={"detail": str(error), "code": error.code})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return workspace.sessions()


@app.post("/sessions/{session_id}/open")
def open_session(session_id: str, meta: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    return workspace.session(session_id).open(meta)


@app.post("/sessions/{session_id}/close")
def close_session(session_id: str) -> dict[str, Any]:
    return workspace.session(session_id).close()


@app.get("/sessions/{session_id}")
def session_info(session_id: str) -> dict[str, Any]:
    return workspace.session(session_id).info()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    return workspace.session(session_id).delete()


@app.post("/sessions/{session_id}/tools/{name}")
def call_tool(session_id: str, name: str, arguments: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return workspace.session(session_id).call(name, arguments)


@app.put("/sessions/{session_id}/files")
async def upload(session_id: str, path: str, request: Request, unique: bool = False) -> dict[str, Any]:
    session = workspace.session(session_id)
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise SandboxError("Plik jest za duży.", 413)
        chunks.append(chunk)
    return session.save_upload(path, chunks, MAX_UPLOAD_BYTES, unique)


@app.get("/sessions/{session_id}/files")
def download(session_id: str, path: str) -> FileResponse:
    session = workspace.session(session_id)
    session.meta()
    return FileResponse(session.existing_file(path))


@app.get("/sessions/{session_id}/preview", response_model=None)
def preview_file(session_id: str, path: str) -> FileResponse | dict[str, Any]:
    session = workspace.session(session_id)
    session.meta()
    try:
        result = build_preview(session.existing_file(path))
    except PreviewError as error:
        raise SandboxError(str(error), 422) from error
    if "file" in result:
        return FileResponse(result["file"], media_type=result["media_type"], headers={"X-Preview-Kind": result["kind"]})
    return result
