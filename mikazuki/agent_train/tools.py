from __future__ import annotations

import json
from typing import Any

from starlette.requests import Request


def make_json_request(payload: dict[str, Any]) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/api/agent-train/tool", "headers": []}, receive)


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {"status": "success", "data": {"result": response}}


class AgentToolExecutor:
    async def execute(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if kind == "run_tagger":
            return await self.run_tagger(payload)
        if kind == "run_differential_lora":
            return await self.run_differential_lora(payload)
        if kind == "run_anima_train":
            return await self.run_anima_train(payload)
        if kind == "run_anima_fast_train":
            return await self.run_anima_fast_train(payload)
        raise ValueError(f"unsupported tool action: {kind}")

    async def run_tagger(self, payload: dict[str, Any]) -> dict[str, Any]:
        from mikazuki.app.tag_edit_leaf_api import run_tagger

        return response_to_dict(await run_tagger(make_json_request(payload)))

    async def run_differential_lora(self, payload: dict[str, Any]) -> dict[str, Any]:
        from mikazuki.app.differential_lora_api import run_differential_training

        return response_to_dict(await run_differential_training(make_json_request(payload)))

    async def run_anima_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        from mikazuki.app.api import create_toml_file

        config = dict(payload)
        config.setdefault("model_train_type", "anima-lora")
        return response_to_dict(await create_toml_file(make_json_request(config)))

    async def run_anima_fast_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        from mikazuki.app.api import create_toml_file

        config = dict(payload)
        config["model_train_type"] = "anima-lora-fast"
        return response_to_dict(await create_toml_file(make_json_request(config)))
