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

    async def subscription_current(self, *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("GET", "/api/subscription/", **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def subscription_upgrade(
        self,
        *,
        plan_id: int,
        months: int = 1,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/api/subscription/upgrade/",
            json={"plan_id": plan_id, "months": months},
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(payload)

    async def create_payment_link(
        self,
        *,
        method: str,
        amount: int,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            "/api/payment/create-transaction/",
            json={"method": method, "amount": amount},
            **_auth_kw(auth_claims),
        )
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

    async def list_env(
        self,
        project_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/enviroment/", params={"project": project_id}, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def create_env(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("POST", "/api/enviroment/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def delete_env(self, env_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/enviroment/{env_id}/", **_auth_kw(auth_claims))

    async def list_releases(
        self,
        project_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/relase/", params={"project": project_id}, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def create_release(
        self,
        project_id: str | int,
        *,
        plan_id: int | None = None,
        months: int | None = None,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"project": project_id}
        if plan_id is not None:
            body["plan_id"] = plan_id
        if months is not None:
            body["months"] = months
        payload = await self._request("POST", "/api/relase/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def list_plans(self, *, auth_claims: dict | None = None) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/plans/", **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def list_files(
        self,
        project_id: str | int | None = None,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        params = {"project": project_id} if project_id is not None else None
        payload = await self._request("GET", "/api/file/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def delete_file(self, file_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/file/{file_id}/", **_auth_kw(auth_claims))

    async def list_collections(
        self,
        project_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", "/api/collection/", params={"project": project_id}, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def get_collection(self, collection_id: str | int, *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("GET", f"/api/collection/{collection_id}/", **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def list_chat_users(
        self,
        project_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", f"/api/chat/users/{project_id}/", **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def user_chat_history(
        self,
        project_id: str | int,
        chat_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", f"/api/chat/messages/{project_id}/{chat_id}/", **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def list_messages(
        self,
        project_id: str | int | None = None,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        params = {"project": project_id} if project_id is not None else None
        payload = await self._request("GET", "/api/message/", params=params, **_auth_kw(auth_claims))
        return _unwrap_list(payload)

    async def create_message(self, body: dict[str, Any], *, auth_claims: dict | None = None) -> dict[str, Any]:
        payload = await self._request("POST", "/api/message/", json=body, **_auth_kw(auth_claims))
        return _unwrap_item(payload)

    async def delete_message(self, message_id: str | int, *, auth_claims: dict | None = None) -> None:
        await self._request("DELETE", f"/api/message/{message_id}/", **_auth_kw(auth_claims))

    def _cms(self, path: str) -> str:
        return f"/api/cms{path}"

    async def cms_list_collections(
        self,
        project_id: str | int,
        *,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        payload = await self._request("GET", self._cms(f"/projects/{project_id}/collections/"), **_auth_kw(auth_claims))
        return payload if isinstance(payload, list) else _unwrap_list(payload)

    async def cms_get_collection(
        self,
        project_id: str | int,
        name: str,
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/"),
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(payload)

    async def cms_create_collection(
        self,
        project_id: str | int,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(payload)

    async def cms_delete_collection(
        self,
        project_id: str | int,
        name: str,
        *,
        auth_claims: dict | None = None,
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_add_field(
        self,
        project_id: str | int,
        name: str,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(payload)

    async def cms_update_field(
        self,
        project_id: str | int,
        name: str,
        field_name: str,
        body: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "PATCH",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/{field_name}/"),
            json=body,
            **_auth_kw(auth_claims),
        )
        return _unwrap_item(payload)

    async def cms_remove_field(
        self,
        project_id: str | int,
        name: str,
        field_name: str,
        *,
        auth_claims: dict | None = None,
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/fields/{field_name}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_list_documents(
        self,
        project_id: str | int,
        name: str,
        *,
        filter: dict[str, Any] | None = None,
        sort: str | None = None,
        limit: int = 50,
        skip: int = 0,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        import json as _json

        params: dict[str, Any] = {"limit": limit, "skip": skip}
        if filter:
            params["filter"] = _json.dumps(filter)
        if sort:
            params["sort"] = sort
        payload = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/"),
            params=params,
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {"results": []}

    async def cms_get_document(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_insert(
        self,
        project_id: str | int,
        name: str,
        document: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/"),
            json=document,
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_bulk_insert(
        self,
        project_id: str | int,
        name: str,
        documents: list[dict[str, Any]],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/bulk/"),
            json={"documents": documents},
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_update(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        patch: dict[str, Any],
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "PATCH",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            json=patch,
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_delete(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        *,
        auth_claims: dict | None = None,
    ) -> None:
        await self._request(
            "DELETE",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/"),
            **_auth_kw(auth_claims),
        )

    async def cms_restore(
        self,
        project_id: str | int,
        name: str,
        document_id: str,
        *,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/{document_id}/restore/"),
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_aggregate(
        self,
        project_id: str | int,
        name: str,
        pipeline: list[dict[str, Any]],
        *,
        limit: int = 500,
        auth_claims: dict | None = None,
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            self._cms(f"/projects/{project_id}/collections/{name}/documents/aggregate/"),
            json={"pipeline": pipeline, "limit": limit},
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, dict) else {}

    async def cms_audit(
        self,
        project_id: str | int,
        name: str,
        *,
        action: str | None = None,
        document_id: str | None = None,
        auth_claims: dict | None = None,
    ) -> list[dict[str, Any]]:
        params = {key: value for key, value in {"action": action, "document_id": document_id}.items() if value}
        payload = await self._request(
            "GET",
            self._cms(f"/projects/{project_id}/collections/{name}/audit/"),
            params=params or None,
            **_auth_kw(auth_claims),
        )
        return payload if isinstance(payload, list) else _unwrap_list(payload)

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
