# Banana Slides - AI Agent Instructions

Banana Slides is an AI-native PowerPoint generation application using nano banana pro🍌 for generating complete presentations from ideas, outlines, or page descriptions. Users can upload reference files, add materials, and modify pages using natural language.

## Architecture Overview

**Stack**: Flask (backend) + React/TypeScript (frontend) + SQLite + Docker
- **Backend**: `backend/` - Flask 3.0 with SQLAlchemy ORM, background task processing via ThreadPoolExecutor
- **Frontend**: `frontend/` - React 18 + Vite + Zustand (state) + TailwindCSS
- **Database**: SQLite with Alembic migrations in `backend/migrations/versions/`
- **AI Services**: Pluggable provider architecture supporting Google Gemini, OpenAI, and Vertex AI

## Core Workflows

### Development Setup
```bash
# Backend: Install with uv (NOT pip)
uv sync                          # Root: installs from pyproject.toml
cd backend && uv run alembic upgrade head  # Run migrations

# Frontend
cd frontend && npm install

# Environment: Copy .env.example to .env in project root
# Both frontend/backend read from root .env via vite.config.ts and app.py
```

### Running Services
```bash
# Development (recommended)
npm run dev                      # Starts docker compose (both services)

# Individual services (for debugging)
npm run dev:backend             # Backend only: uv run python app.py
npm run dev:frontend            # Frontend only: npm run dev (in frontend/)

# Port configuration: Set PORT in root .env (default 5000)
# Frontend proxy auto-adjusts in vite.config.ts
```

### Testing Strategy
```bash
# Quick tests (no service needed)
npm run test:backend            # uv run pytest backend/tests/ -v
npm run test:frontend           # cd frontend && npm test -- --run

# E2E tests (requires running service)
npm run test:e2e                # Frontend Playwright tests
npm run test:docker             # Full Docker environment test

# Mock AI: Set USE_MOCK_AI=true in tests (see backend/tests/conftest.py)
```

## Key Architectural Patterns

### 1. AI Provider Abstraction (`backend/services/ai_providers/`)
- **Factory pattern**: `get_text_provider(model)` and `get_image_provider(model)` create providers based on `AI_PROVIDER_FORMAT` env var
- **Base classes**: `TextProvider` and `ImageProvider` in `text/base.py` and `image/base.py`
- **Implementations**: `GenAITextProvider`, `OpenAITextProvider`, `GenAIImageProvider`, `OpenAIImageProvider`
- **Singleton caching**: `ai_service_manager.py` maintains provider cache to avoid re-initialization overhead

### 2. Background Task Management (`backend/services/task_manager.py`)
- Uses `ThreadPoolExecutor` (no Celery/Redis needed)
- Task tracking via SQLAlchemy `Task` model with status updates
- Pattern: Controllers submit tasks via `task_manager.submit_task(task_id, func, *args)`, frontend polls `/api/tasks/{task_id}`
- Example: `generate_descriptions_task()` and `generate_images_task()` are typical background functions

### 3. State Management (Frontend)
- **Zustand store**: `frontend/src/store/useProjectStore.ts` is the single source of truth
- **Debounced updates**: `debouncedUpdatePage()` batches API calls (1000ms debounce)
- **Task polling**: `pollTask()` and `pollImageTask()` handle async operations
- **Normalization**: `normalizeProject()` in `utils/index.ts` ensures consistent data shape

### 4. Database Migrations (Alembic)
- **Never use `db.create_all()`** - Always use Alembic migrations
- New models: Add to `models/`, then `cd backend && uv run alembic revision --autogenerate -m "description"`
- Apply: `uv run alembic upgrade head`
- Migrations must be **idempotent** (see `001_baseline_schema.py` for pattern checking existing tables)

### 5. Response Format Convention
All API responses use `utils/response.py` helpers:
```python
from utils import success_response, error_response, not_found, bad_request

# Success: {"success": true, "data": {...}, "message": "..."}
return success_response(data=result, message="Created")

# Error: {"success": false, "error_code": "...", "message": "..."}
return error_response("VALIDATION_ERROR", "Invalid input", 400)
```

### 6. Project Creation Types
Three entry points defined in `creation_type` field:
- `idea`: User provides single sentence → AI generates outline → descriptions → images
- `outline`: User provides structured outline → AI generates descriptions → images  
- `description`: User provides per-page descriptions → AI generates images directly

Flow controlled in `backend/controllers/project_controller.py` and `frontend/src/pages/`

## File Organization Conventions

### Backend Structure
- `controllers/`: Flask blueprints (route handlers) - thin layer, delegate to services
- `services/`: Business logic (AI operations, file parsing, export) - thick layer
- `models/`: SQLAlchemy models (Project, Page, Task, Material, ReferenceFile, etc.)
- `utils/`: Pure functions (response helpers, validators, path utilities, pptx_builder)
- `migrations/versions/`: Alembic migrations (numbered: `001_`, `002_`, etc.)

### Frontend Structure  
- `pages/`: Route components (Home, OutlineEditor, DetailEditor, SlidePreview)
- `components/`: Reusable UI (split into `shared/`, `outline/`, `preview/`)
- `store/`: Zustand state (only `useProjectStore.ts` currently)
- `api/`: API client (`client.ts` for axios config, `endpoints.ts` for typed calls)
- `e2e/`: Playwright tests (mocked: `ui-full-flow-mocked.spec.ts`, real: `ui-full-flow.spec.ts`)

## Critical Implementation Details

### Configuration Precedence
1. **Root `.env`** is loaded by both services (not `.env` files in subdirectories)
2. Backend: `app.py` uses `load_dotenv(dotenv_path=_project_root / '.env')`
3. Frontend: `vite.config.ts` sets `envDir: path.resolve(__dirname, '..')`
4. Database settings: Stored in `settings` table, override env vars at runtime (see `backend/models/settings.py`)

### File Upload Paths
- **Upload folder**: `PROJECT_ROOT/uploads/` (shared between Docker and local dev)
- **Storage pattern**: `uploads/{project_id}/materials/` and `uploads/{project_id}/reference_files/`
- **Path utilities**: Use `backend/utils/path_utils.py` functions to construct paths
- **URL serving**: Backend serves files at `/files/<path:filename>` route

### Material vs Reference Files
- **Materials** (`models/material.py`): User-uploaded images for page generation context (can be project-specific or global)
- **Reference Files** (`models/reference_file.py`): Documents (PDF, DOCX, etc.) parsed by MinerU service into markdown, stored in `markdown_content` field
- **Parsing**: `backend/services/file_parser_service.py` handles MinerU integration and optional image captioning

### Page Image Versioning
- `PageImageVersion` model tracks editing history (see `models/page_image_version.py`)
- Helper: `save_image_with_version()` in `task_manager.py` creates history entry when generating/editing
- Frontend shows version history in preview page

### Export Mechanisms
Three export types in `backend/services/export_service.py`:
1. **Standard PPTX**: Uses `python-pptx`, embeds PNG images directly
2. **PDF**: Uses `img2pdf` library for high-quality conversion
3. **Editable PPTX** (Beta): `utils/pptx_builder.py` uses image analysis to extract text/shapes, creating editable elements

## Testing Patterns

### Backend Test Fixtures (`backend/tests/conftest.py`)
- `app`: Flask app factory with test config
- `client`: Test client for API calls (no service needed)
- `db_session`: Database session with automatic cleanup
- `sample_project`: Pre-created project for tests
- **Mock AI**: Set `USE_MOCK_AI=true` env var to use mock responses

### Frontend E2E Tests
- **Mocked** (`ui-full-flow-mocked.spec.ts`): Route interception for fast CI (1-2 min)
- **Real** (`ui-full-flow.spec.ts`): Actual API calls with real service (10+ min)
- Use `@playwright/test` fixtures, timeout extended for AI operations

### Integration Test Markers
- `@pytest.mark.requires_service`: Skipped in CI unless Docker is running
- See `backend/tests/integration/README.md` for CI strategy

## Docker Specifics

### Build Args (Windows/WSL compatibility)
- `DOCKER_BUILDKIT=1` and `DOCKER_REGISTRY` args in docker-compose.yml
- Frontend build uses `--no-follow-symlinks` to avoid Windows path issues

### Health Checks
- Backend: `curl -f http://localhost:${PORT:-5000}/health`
- Used by `scripts/wait-for-health.sh` in test workflows

### Volume Mounts
- `./backend/instance:/app/backend/instance` - SQLite database persistence
- `./uploads:/app/uploads` - File uploads persistence

## Common Pitfalls

1. **Don't call `db.create_all()`** - Use Alembic migrations instead
2. **Don't install packages with pip** - Use `uv sync` or `uv add package` to update pyproject.toml
3. **Don't create `.env` in subdirectories** - Only root `.env` is used
4. **Don't use `&&` in PowerShell** - Use `;` to chain commands (Windows default shell)
5. **Don't forget `uv run`** - Prefix all Python commands in scripts (e.g., `uv run pytest`, not `pytest`)
6. **Don't hardcode API base URLs** - Use relative paths, proxy handles routing (see `frontend/src/api/client.ts`)

## Script Utilities

- `scripts/setup_git_hooks.sh`: Pre-push hooks for lint/test
- `scripts/test_docker_environment.sh`: Full Docker environment validation
- `scripts/wait-for-health.sh`: Health check polling for CI
- `scripts/run-local-ci.sh`: Run full CI pipeline locally

When modifying core architecture (AI providers, task system, database schema), ensure updates are reflected in corresponding tests and documentation.
