# 数据集打标 API（Phase 1）

> 打标页进度条与后台任务状态。完整 #40 范围见 `docs/issues/40-dataset-tagging-backend-alignment.md`。

## GET `/api/tagger/status`

轮询打标/下载任务状态。

**响应示例**

```json
{
  "status": "success",
  "message": null,
  "data": {
    "phase": "tagging",
    "message": "正在打标 foo.jpg (3/128)",
    "model": "wd14-convnextv2-v2",
    "download": { "current": 0, "total": 0, "filename": "" },
    "tagging": { "current": 3, "total": 128, "filename": "foo.jpg" },
    "error": null,
    "updated_at": 1710000000.0
  }
}
```

`phase`：`idle` | `downloading` | `tagging` | `cancelling` | `done` | `error`

## GET `/api/tagger/download-status`

仅返回下载相关字段（与 `status.download` 一致，便于 #40 对齐）。

## POST `/api/tagger/reset`

清空打标/下载进度状态（`phase` → `idle`）。若任务仍在进行会先请求取消。打标页「全部重置」会调用此接口。

## POST `/api/tagger/cancel`

中止当前进行中的模型下载或打标任务（协作式取消：在下载块/每张图打标前检查）。

**成功**：`status: success`，`message: 正在中止任务…` 或 `当前无运行中的任务`  
中止完成后 `GET /api/tagger/status` 的 `phase` 为 `idle`，`message` 为 `已中止`。

## POST `/api/tagger/prefetch`

后台下载所选 HF 模型文件（按 interrogator 的 `model.onnx` + 标签文件，分步更新 `download.current/total`）。

打标页「预下载」在下载进行中会变为「中止」，与「启动」共用 `POST /api/tagger/cancel`。

**请求体**

```json
{ "interrogator_model": "wd14-convnextv2-v2" }
```

**成功**：`status: success`，`message: 模型下载已开始`  
**失败**：已有任务进行中、未知模型等返回 `status: fail`。

## POST `/api/interrogate`

与现网兼容；提交后由后台任务更新 `tagging` 进度。若已有下载/打标任务则返回失败。

请求体字段见 `TaggerInterrogateRequest`（`mikazuki/app/models.py`）。
