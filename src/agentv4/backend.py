from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from .config import Settings, get_settings


class BackendError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.details = details


def _unwrap_item(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _unwrap_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list):
        return results
    if isinstance(payload.get("results"), list):
        return payload["results"]
    return []


def _auth_kw(auth_claims: dict | None) -> dict[str, Any]:
    if not auth_claims:
        return {}
    return {
        "bearer": auth_claims.get("access_token"),
        "refresh_token": auth_claims.get("refresh_token"),
    }


class BackendClient:
    def __init__(self, base_url: str, *, internal_token: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._internal_token = internal_token
        self._client = httpx.AsyncClient(timeout=timeout)

    def _headers(self, *, bearer: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        elif self._internal_token:
            headers["X-Internal-Token"] = self._internal_token
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _try_refresh(self, refresh_token: str | None) -> str | None:
        if not refresh_token:
            return None
        try:
            response = await self._client.post(
                f"{self._base_url}/auth/token/refresh/",
                json={"refresh": refresh_token},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
            if response.status_code >= 400:
                return None
            payload = response.json() or {}
            root = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            token = (root or {}).get("token") or {}
            access = token.get("access") or root.get("access")
            return access if isinstance(access, str) and access else None
        except Exception:
            return None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        bearer: str | None = None,
        refresh_token: str | None = None,
    ) -> Any:
        url = self._base_url + path
        response = await self._client.request(
            method,
            url,
            params=params,
            json=json,
            headers=self._headers(bearer=bearer),
        )
        if response.status_code == 401 and refresh_token:
            new_access = await self._try_refresh(refresh_token)
            if new_access:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=self._headers(bearer=new_access),
                )
        if response.status_code >= 400:
            try:
                details = response.json()
            except Exception:
                details = response.text
            raise BackendError(f"{method} {path} -> {response.status_code}", status=response.status_code, details=details)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def subscription_usage(self, *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("GET", "/api/subscription/usage/", **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def list_projects(
        self,
        *,
        search: str | None = None,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        params = {"search": search} if search else None
        payload = await self._request("GET", "/api/project/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def get_project(self, project_id: str | int, *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("GET", f"/api/project/{project_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def create_project(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("POST", "/api/project/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def update_project(
        self,
        project_id: str | int,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request("PATCH", f"/api/project/{project_id}/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def delete_project(self, project_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/project/{project_id}/", **_auth_kw(auth_claims))

    async def report_token_usage(
        self,
        *,
        user_id: str | int,
        input_tokens: int,
        output_tokens: int,
        project_id: str | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "user_id": int(user_id),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if project_id:
            body["project_id"] = project_id
        await self._request("POST", "/api/subscription/usage/report/", json=body)


@lru_cache(maxsize=1)
def get_backend_client(settings: Settings | None = None) -> BackendClient:
    s = settings or get_settings()
    return BackendClient(
        s.backend_base_url,
        internal_token=s.backend_internal_token,
        timeout=s.backend_timeout,
    )
