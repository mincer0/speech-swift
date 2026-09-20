# MiniCPM Demo service layer

This layer mirrors the pinned upstream Demo gateway (`ba7fa9cc6ad63c894f1bd5e5afac28466953519d`) without coupling routes to model code.

| Upstream route/operation | Swift service API |
| --- | --- |
| `session.init`, `session.created`, `GET /api/sessions` | `MiniCPMDemoSessionStore.create`, `metadata`, `list` |
| JSONL stream recording | `MiniCPMDemoSessionStore.appendEvent` / `MiniCPMDemoRecordingService.append` |
| PCM/JPEG/video session blobs | `storeRecordingBlob` / `MiniCPMDemoRecordingService.storeBlob` |
| `GET /api/sessions/{id}/recording` | `SessionStore.recording` |
| secure session asset download | `RecordingSnapshot.blobs`, `AssetStore.download` |
| `POST/GET /api/sessions/{id}/comment` | `MiniCPMDemoShareStore.setComment`, `comment` |
| share links / revoke / expiry | `MiniCPMDemoShareStore.createShare`, `resolveShare`, `revokeShare` |
| `GET /api/apps`, `PUT /api/admin/apps/{id}` | `MiniCPMDemoAdminService.listApps`, `setAppEnabled` |
| `GET /api/assets/ref_audio` + CRUD | `MiniCPMDemoAssetStore.list`, `uploadReferenceAudio`, `download`, `delete` |
| `GET /api/frontend_defaults` | `MiniCPMDemoConfigStore.frontendDefaults` |
| `GET/PUT /api/config/eta` + EMA | `MiniCPMDemoConfigStore.etaStatus`, `updateETA`, `recordDuration` |
| `GET /api/presets`, preset audio | `MiniCPMDemoPresetStore.list`, `audio` |
| `GET /api/default_ref_audio` | `MiniCPMDemoAssetStore.setDefaultReferenceAudio`, `defaultReferenceAudio` |

## Storage and lifecycle

The root contains `sessions/<session_id>/`, `assets/assets/`, and `admin/`. Session metadata is `meta.json`; event frames are append-only `recording/stream.jsonl`; large media is stored as unique files under `recording/blob/`. Reference assets use `assets/assets/registry.json`. Comments and share hashes are persisted under the session/admin directories.

All services are actors. IDs, extensions and preset paths are validated before filesystem access; resolved paths must remain below their configured root. Session creation rejects an existing directory, and blob/asset writes use generated names and an existence check rather than client filenames.

`MiniCPMDemoStorageLimits` enforces per-blob, per-session, per-event and process-wide byte limits before writes. `MiniCPMDemoSessionStore.cleanup` removes only closed sessions by retention or oldest-first capacity; active sessions are never deleted. `MiniCPMDemoAdminService.cleanup` also removes stale asset registry rows.

The route layer should inject one shared `MiniCPMDemoStorageLayout`, session store, asset store, share store, config store and admin service. It should stream `AssetDownload.data`/recording blobs directly and must not reconstruct a whole session in memory for normal event recording.
