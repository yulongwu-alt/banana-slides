from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import math
from pydantic import BaseModel, Field,field_validator
import os
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
from urllib.parse import quote, urlencode
from typing import Any, Dict, Optional
import logging
from src.mlaas.schema import SCHEMA
from google import genai
from openai import AsyncOpenAI
import base64
from google.genai.types import GenerateContentConfig, Modality
import google.genai.types as types
from PIL import Image
from io import BytesIO
from src.mlaas.telemetry import Telemetry
from src.mlaas.sagemaker import check_backlog, generate_prompts_for_items, submit_batch_jobs, ICON_SYSTEM_PROMPT
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.config import Config
import asyncio
import uuid

logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger("Info")

class HealthCheckAccessFilter(logging.Filter):
    def filter(self, record):
        args = getattr(record, "args", ())
        if len(args) >= 3:
            request_path = str(args[2]).split("?", 1)[0]
            if request_path == "/health":
                return False
        return True

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckAccessFilter())



class ImageGenerationInput(BaseModel):
    user_prompt: str  = Field(..., max_length=2000)
    base64_images: list[str] = Field(..., max_items=5)
    prompt_enhance: bool = True
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate concurrently")

    @field_validator('base64_images')
    def validate_image_size(cls, v):
        for img in v:
            if len(img) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image too large")
        return v

class ImageGenerationResult(BaseModel):
    base64_images: list[str] = Field(..., description="List of generated images in base64 format")

class WEditInput(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    num_inference_steps: int = Field(default=40, ge=1, le=100)
    true_cfg_scale: float = Field(default=4.0, ge=0.0, le=20.0)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    num_images_per_prompt: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default=-1)
    lora_name: str = Field(default="char", pattern="^(char|item|spine|vic-item-icon)$")

    @field_validator('image')
    def validate_image_size(cls, v):
        if len(v) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("Image too large")
        return v
    
    @field_validator('lora_name')
    def validate_lora_name(cls, v):
        if v not in ['char', 'item', 'spine', 'vic-item-icon']:
            raise ValueError('lora_name must be either "char", "item", "spine", or "vic-item-icon"')
        return v

class WEditResult(BaseModel):
    images_base64: list[str]
    meta: dict = {}

class BatchJobInput(BaseModel):
    keywords: list[str]

    space: str = Field(..., pattern="^(public|private)$")#for future use
    lora_name: str = Field(default="props", pattern="^(props|icons)$")
    
    @field_validator('space')
    def validate_space(cls, v):
        if v not in ['public', 'private']:
            raise ValueError("space must be either 'public' or 'private'")
        return v

    @field_validator('lora_name')
    def validate_lora_name(cls, v):
        if v not in ['props', 'icons']:
            raise ValueError("lora_name must be either 'props' or 'icons'")
        return v

class BatchJobStatus(BaseModel):
    status: str
    message: str
    job_count: int = 0

class WT2IInput(BaseModel):
    keyword: str = Field(..., max_length=500, description="Single keyword for image generation")
    lora_name: str = Field(default="props", pattern="^(props|icons)$")
    num_inference_steps: int = Field(default=40, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)
    num_images_per_prompt: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default=-1)
    
    @field_validator('lora_name')
    def validate_lora_name(cls, v):
        if v not in ['props', 'icons']:
            raise ValueError("lora_name must be either 'props' or 'icons'")
        return v

class WT2IResult(BaseModel):
    images_base64: list[str]
    prompt: str
    meta: dict = {}

class TokenExchangeRequest(BaseModel):
    code: str
    redirect_uri: str

class TokenRefreshRequest(BaseModel):
    refresh_token: str


class FeishuAuthError(RuntimeError):
    """Raised when Feishu auth API returns an error."""

    def __init__(self, code: int, msg: str, response: Optional[Dict[str, Any]] = None):
        self.code = code
        self.msg = msg
        self.response = dict(response or {})
        super().__init__(f"Feishu auth error {code}: {msg}")


class FeishuLoginResponse(BaseModel):
    login_url: str


class FeishuExchangeCodeRequest(BaseModel):
    code: str = Field(..., description="Code returned by Feishu after login")
    redirect_uri: str
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    timeout: int = 30


class FeishuRefreshTokenRequest(BaseModel):
    refresh_token: str
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    timeout: int = 30


AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "")
DATABASE_DSN = os.getenv("DATABASE_DSN", "")

SSO_AUTHORIZE_URL = os.getenv("SSO_AUTHORIZE_URL", "")
SSO_TOKEN_URL = os.getenv("SSO_TOKEN_URL", "")
SSO_CLIENT_ID = os.getenv("SSO_CLIENT_ID", "")
SSO_CLIENT_SECRET = os.getenv("SSO_CLIENT_SECRET", "")
FEISHU_BASE_URL = "https://open.feishu.cn"
FEISHU_OPEN_API_BASE_URL = f"{FEISHU_BASE_URL}/open-apis"

BASE_URL = os.getenv("LITELLM_BASE_URL", "") 
LITELLM_TOKEN =  os.getenv("LITELLM_TOKEN", "") 
VERTEXAI_TOKEN = os.getenv("VERTEXAI_TOKEN", "")
NANOBANANA_API_KEY = os.getenv("NANOBANANA_API_KEY", "")
NANOBANANA_API_KEY_BACKUP = os.getenv("NANOBANANA_API_KEY_BACKUP", "")
GENAI_BASE_URL = os.getenv("GENAI_BASE_URL", "")
GENAI_BASE_URL_BACKUP = os.getenv("GENAI_BASE_URL_BACKUP", "")
SEEDREAM_BASE_URL = os.getenv("SEEDREAM_BASE_URL", "")
SEEDREAM_API_KEY = os.getenv("SEEDREAM_API_KEY", "")
MULTI_IMAGE_WHITELIST = os.getenv("MULTI_IMAGE_WHITELIST", "")

# Vertex AI configuration for backup
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT", "aigc-405003")
VERTEX_REGION = os.getenv("VERTEX_REGION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID", "512547240892432384")


SEEDREAM_SYTEM_PROMPT = '''
你是「Seedream 4.0/4.5 提示词生成与优化助手」。你的唯一目标：把用户的想法/草稿 prompt，改写成符合 Seedream 最佳实践、可直接粘贴使用的高质量提示词。

工作原则（必须遵守）
1) 用自然语言清晰描述画面：优先“主体 + 行为/状态 + 环境/场景”，再补充风格、色彩、光影、构图、镜头等美学要素。避免关键词堆叠式逗号罗列。
2) 简洁精确优先：不要为了“看起来高级”而重复堆砌同义词、华丽形容词或无效咒语；确保每一句都能改变画面。
3) 明确用途/类型：如果是 logo、海报、信息图、UI、分镜、表情包、产品图等，必须在 prompt 里写清“要做什么类型的图、用于什么场景”。
4) 文字渲染：凡是要出现在图里的文字，一律用英文双引号包起来，保持字面内容完全一致（大小写/空格/标点/换行都要保留）。
5) 图片编辑/图生图：指令要“简洁明确”，明确要改谁、改什么、怎么改；避免“它/这个/那个”等模糊代词。若除修改点外都要保持不变，必须显式写“其余保持不变/动作表情不变/布局不变”等。
6) 参考图生图：提示词必须包含两部分：
   A. 指明参考对象：要从参考图保留什么（人物形象/产品材质/视觉风格/布局等）
   B. 描述生成画面：你要生成的新场景与内容细节
7) 多图输入：当用户提供多张图时，用“图一/图二/图三…”明确各图分别用于替换/组合/风格迁移/参考哪个对象。
8) 多图输出（组图生成）：当用户需要一套/一系列图时，用“一套/组图/一系列”或直接写明数量（如“生成四张图/生成七张图”），并强调“角色连贯/风格统一/同一视觉体系”。

你要做的事（流程）
Step 0 识别任务类型（自动判断，不要反问）：
- 文生图 / 图片编辑 / 参考图生图 / 多图输入复合编辑 / 多图输出组图
Step 1 抽取用户意图与约束：
- 主体、行为、环境、用途/类型、风格、构图/镜头、光影、色彩、材质、细节清单、需要出现的文字、必须保持不变的元素
Step 2 生成“最终可用提示词”：
- 用 1–4 句自然语言写完：先核心画面，再补充美学与约束
- 需要文字就加引号；需要“保持不变”就直说；需要对照多图就标注图一/图二…
Step 3 根据用户的需求给出更简洁或更细节的版本


输出格式（严格遵守）
只输出以下结构，不要输出多余解释，不要使用 Markdown：
一段可直接用于 Seedream 的 prompt，仅返回prompt
'''

WItem_EDIT_POSITIVE_PROMPT = '''
wuxia-icon-item, Using the provided item icon as the base, KEEP the exact silhouette, proportions, pose, and all original design details.
Unify the rendering into a consistent hand-painted  game icon style:
- clean readable forms, medium contrast
- consistent rough ink/charcoal outer outline (same thickness everywhere), slightly dry-brush edge
- soft painterly shading with subtle paper grain, no photorealism
- centered composition on pure white background
Do NOT add new accessories, symbols, text, or change colors dramatically. Only harmonize style, edges, lighting, outline. 
'''




VICItem_EDIT_POSITIVE_PROMPT = '''
vic-item-icon, using the provided item icon as the **content/base**, KEEP the exact silhouette, proportions, angle/pose, and all original design details.

Restyle it to match the vintage hand-painted collectible icon look:

* **vintage hand-painted collectible icon** look, but **controlled and crisp**
* **ink-and-gouache / dry-brush illustration** feel (NOT heavy watercolor wash)
* warm, slightly aged palette (cream / parchment / leather / brass / muted earthy tones)
* hand-drawn ink contour lines with natural taper and slight wobble, clear readable edges
* painterly fills with **tight brush control**, **minimal bleed**, **minimal watercolor bloom**
* soft shading with clean form separation; keep values readable for game icon clarity
* subtle surface texture only (very light paper grain), avoid obvious wet-paper stains
* materials should feel illustrated and polished (wood, brass, glass, paper), not loose sketchy washes
* centered composition on **pure white background**


**Do NOT** add new accessories, symbols, labels, stamps, text, or decorative elements.
**Do NOT** change the object design, silhouette, proportions, or major colors dramatically.
Only change the rendering style: line quality, brushwork control, texture intensity, shading, and lighting harmony to match the uploaded reference while keeping the result cleaner and less watercolor-like.

'''

VICItem_EDIT_NEGATIVE_PROMPT = '''
photorealistic, 3D render, CGI, ultra-detailed realism, glossy plastic look, hard specular highlights, metallic reflections that overpower form, high contrast dramatic lighting, neon colors, oversaturated colors, cold blue color cast, grayscale, flat unshaded color blocks, vector art, cel shading, anime style, pixel art, low-poly, cartoon simplification, messy composition, off-center object, cropped object, tilted canvas, background scene, colored background, gray background, shadow-heavy backdrop, clutter, extra props, extra accessories, added symbols, added ornaments, added labels, added text, watermark, logo, stamps, UI frame, border, duplicate objects, altered silhouette, changed proportions, changed pose, changed angle, design changes, missing parts, warped geometry, deformed shape, inconsistent line thickness, clean digital outline, overly smooth airbrush, noisy texture, heavy grunge, dirty smudges, blurry image, low resolution, jagged edges, compression artifacts,
heavy watercolor wash, watercolor bloom, wet-on-wet texture, strong pigment bleed, paper stains, ink spread, washed-out edges, overly soft mushy shading, loose sketch wash, excessive paper grain, blotchy watercolor texture, faded low-contrast watercolor illustration
'''

WSpine_EDIT_POSITIVE_PROMPT = "wuxia-spine-v1, "



WSpine_EDIT_NEGATIVE_PROMPT = '''
text, watermark, logo,photorealistic, 3D render, cinematic lighting, hard rim neon glow, thick comic lineart, vector flat icon, pixel art,distorted shape,
over-sharpen, noisy artifacts, plastic material look
'''

WItem_EDIT_NEGATIVE_PROMPT = '''
photorealistic, 3D render, cinematic lighting, hard rim neon glow, thick comic lineart, vector flat icon, pixel art,
busy background, gradient background, vignette, text, watermark, logo, extra objects, cropped item, distorted shape,
over-sharpen, noisy artifacts, plastic material look
'''


WCharEdit_SYSTEM_PROMPT = '''
You are an assistant that writes English image-edit prompts to restyle pictures into a unified 2D ancient Chinese character illustration style (“古风立绘风”).

You MUST:
- Always output a single JSON object with exactly two keys: "positive_prompt" and "negative_prompt".
- Always follow the fixed templates below.
- Keep all non-bracket text in the templates EXACTLY as written.
- ONLY replace the parts inside [BRACKETS] with short English phrases based on the user’s description.

Your final answer MUST be valid JSON, with this structure:

{
  "positive_prompt": "...",
  "negative_prompt": "..."
}

No extra keys, no explanations, no markdown.

---

TEMPLATE FOR positive_prompt
You MUST use this template text verbatim, only filling in the bracketed placeholders:

"Keep the original [SUBJECT_AND_PRESERVE] without changing the overall silhouette or composition.

Restyle the image into our unified 2D ancient Chinese character illustration style: clean crisp ink lineart with dark sepia/black outlines, slightly varied line weight and a subtle hand-drawn ink-brush feeling. Simplify the rendering into mostly flat colors with gentle cel shading (about 2–3 tones per area) and soft gradients on the clothing and hair. Remove realistic fabric lighting and specular highlights, instead use clear hand-painted folds and a light stylized cloth texture to suggest material. Soften the facial rendering to a semi-anime look with clear contour lines around the eyes and nose, smooth skin tones, and a [EXPRESSION_AND_MOOD] expression that fits the character personality.

Use a plain clean white or very light warm gray background with only a soft cast shadow under and behind the character, no detailed scene or extra props. Make the final image look like a polished 2D ancient Chinese game character illustration consistent with our other artworks, not like a realistic painting or 3D render. [OPTIONAL_EXTRA_INSTRUCTIONS]"

Where you MUST:
- Replace [SUBJECT_AND_PRESERVE] with a concise description of what must stay the same (e.g. "character design, clothing, hairstyle, accessories, proportions and seated pose of the young Chinese scholar").
- Replace [EXPRESSION_AND_MOOD] with a short mood phrase (e.g. "calm and slightly melancholic", "gentle and warm", "serious and focused").
- Replace [OPTIONAL_EXTRA_INSTRUCTIONS] with either:
  - a short extra clause if the user asked for specific tweaks (e.g. "Keep the existing color palette and main costume details."), OR
  - nothing (remove the brackets and this placeholder entirely) if there is nothing extra.

Do NOT change any other words in the template.

---

TEMPLATE FOR negative_prompt
You MUST use this template text verbatim, only adding optional user-specific negatives at the end:

"photorealistic, 3D render, CGI, hyper-realistic, ultra-detailed skin, pores, subsurface scattering, strong specular highlights, harsh dramatic lighting, bloom, lens flare, depth of field, bokeh, noisy background, complex environment, realistic scenery, extra characters, crowd, duplicate limbs, distorted anatomy, broken hands, disfigured face, low resolution, blur, pixelation, compression artifacts, watermark, logo, UI, text, subtitles[OPTIONAL_USER_SPECIFIC_NEGATIVES]"

Where you MUST:
- Replace [OPTIONAL_USER_SPECIFIC_NEGATIVES] with:
  - a comma + any extra things the user explicitly does NOT want (e.g. ", blood, gore, armor, tattoos"), OR
  - nothing at all (delete the brackets) if the user has no additional negatives.

Do NOT change any other words in the template.

---

OUTPUT RULES (CRITICAL)
- Always output ONLY a single JSON object, for example:

{
  "positive_prompt": "Keep the original ...",
  "negative_prompt": "photorealistic, 3D render, ..."
}

- Do NOT include explanations, markdown, or any text outside this JSON.
- Do NOT mention the templates or placeholders in the output.
- All generated English must be natural and fluent.
'''


WCharEdit_OUTLINE_ENHANCE_SYSTEM_PROMPT ='''
使外轮廓更厚（中式勾线），笔触更有中国风，保持其他部分不变
'''

if not all([LITELLM_TOKEN, VERTEXAI_TOKEN, NANOBANANA_API_KEY]):
    raise ValueError("Missing required API keys in environment variables")

# Global async resources
db_pool = None
http_client = None
s3_client = None
s3_client_us_east_2 = None
client_chat = None
client_seedream = None
client = None
client_backup = None
w_edit_lock = None
w_t2i_lock = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    global db_pool, http_client, s3_client, s3_client_us_east_2, client_chat, client_seedream, client, client_backup, w_edit_lock, w_t2i_lock
    
    # Initialize the lock for w-edit endpoint
    w_edit_lock = asyncio.Lock()

    # Initialize the lock for w-t2i endpoint
    w_t2i_lock = asyncio.Lock()

    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "")
    s3_endpoint_ohio = os.getenv("S3_ENDPOINT_URL_OHIO", "")
    if s3_endpoint_url:
        
        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
    else:
        s3_client = boto3.client('s3')
       
    

    if s3_endpoint_ohio:
        s3_client_us_east_2 = boto3.client(
            's3',
            region_name='us-east-2',
            endpoint_url=s3_endpoint_ohio,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test")
        )
    else:
        s3_client_us_east_2 = boto3.client('s3', region_name='us-east-2')


    db_pool = AsyncConnectionPool(DATABASE_DSN, min_size=5, max_size=20)
    await db_pool.open()
    http_client = httpx.AsyncClient()
    client_chat = AsyncOpenAI(
        api_key=LITELLM_TOKEN,
        base_url=BASE_URL
    )

    client_seedream = AsyncOpenAI(
        api_key=SEEDREAM_API_KEY,
        base_url=SEEDREAM_BASE_URL
    )

    
    client = genai.Client(api_key=NANOBANANA_API_KEY)

    if NANOBANANA_API_KEY_BACKUP:
        
        client_backup =  genai.Client(api_key=NANOBANANA_API_KEY_BACKUP,http_options=types.HttpOptions(base_url=GENAI_BASE_URL_BACKUP))
    else:
        client_backup = None
    
    # Initialize database schema
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            for table_def in SCHEMA:
                table_name = table_def["table"]
                
                # Check if table exists
                await cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                    (table_name,)
                )
                table_exists = await cur.fetchone()
                
                if not table_exists[0]:
                    logger.info(f"Creating table {table_name}...")
                    
                    # Build CREATE TABLE statement
                    columns_sql = []
                    
                    # Add auto-incremental primary key
                    columns_sql.append("id SERIAL PRIMARY KEY")
                    
                    # Add schema columns
                    for col in table_def["columns"]:
                        col_sql = f"{col['name']} {col['type']}"
                        
                        if col.get("primary_key"):
                            col_sql += " PRIMARY KEY"
                        
                        if not col.get("nullable"):
                            col_sql += " NOT NULL"
                        
                        if col.get("default"):
                            col_sql += f" DEFAULT {col['default']}"
                        
                        columns_sql.append(col_sql)
                    
                    # Add created_at timestamp
                    columns_sql.append("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL")
                    
                    create_table_sql = f"CREATE TABLE {table_name} ({', '.join(columns_sql)})"
                    await cur.execute(create_table_sql)
                    logger.info(f"Table {table_name} created successfully")
                else:
                    # Check for missing columns
                    await cur.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                        (table_name,)
                    )
                    existing_columns = [row[0] for row in await cur.fetchall()]
                    
                    for col in table_def["columns"]:
                        if col["name"] not in existing_columns:
                            logger.info(f"Adding column {col['name']} to table {table_name}...")
                            col_sql = f"ALTER TABLE {table_name} ADD COLUMN {col['name']} {col['type']}"
                            
                            if col.get("primary_key"):
                                col_sql += " PRIMARY KEY"
                            
                            if not col.get("nullable"):
                                col_sql += " NOT NULL"
                            
                            if col.get("default"):
                                col_sql += f" DEFAULT {col['default']}"
                                
                            await cur.execute(col_sql)
                            logger.info(f"Column {col['name']} added successfully")
            
            await conn.commit()
    
    yield
    
    # Shutdown No need to Check
    if client is not None:
        await client.aio.aclose()
        client = None
    if client_backup is not None:
        await client_backup.aio.aclose()
        client_backup = None
    if client_chat is not None:
        await client_chat.close()
        client_chat = None
    if client_seedream is not None:
        await client_seedream.close()
        client_seedream = None
    if http_client is not None:
        await http_client.aclose()
        http_client = None
    
    if db_pool is not None:
        await db_pool.close()
        db_pool = None
    if s3_client is not None:
        s3_client = None
    if s3_client_us_east_2 is not None:
        s3_client_us_east_2 = None


def _require_feishu_app_id(value: Optional[str]) -> str:
    app_id = value or os.getenv("FEISHU_APP_ID")
    if not app_id:
        raise ValueError("app_id is required. Pass it in the request or set FEISHU_APP_ID.")
    return app_id


def _require_feishu_app_secret(value: Optional[str]) -> str:
    app_secret = value or os.getenv("FEISHU_APP_SECRET")
    if not app_secret:
        raise ValueError(
            "app_secret is required. Pass it in the request or set FEISHU_APP_SECRET."
        )
    return app_secret


def build_feishu_login_url(
    app_id: str,
    redirect_uri: str,
    *,
    state: Optional[str] = None,
) -> str:
    if not app_id or not app_id.strip():
        raise ValueError("app_id is required")
    if not redirect_uri or not redirect_uri.strip():
        raise ValueError("redirect_uri is required")

    query = {
        "app_id": app_id.strip(),
        "redirect_uri": redirect_uri.strip(),
    }
    if state:
        query["state"] = state
    return f"{FEISHU_BASE_URL}/open-apis/authen/v1/index?{urlencode(query)}"


async def exchange_feishu_code_for_user_token(
    *,
    app_id: str,
    app_secret: str,
    code: str,
    redirect_uri: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    app_access_token = await get_feishu_app_access_token(
        app_id=app_id,
        app_secret=app_secret,
        timeout=timeout,
    )
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    return await _post_feishu_auth_json(
        "/authen/v1/oidc/access_token",
        payload=payload,
        timeout=timeout,
        app_access_token=app_access_token,
    )


async def refresh_feishu_user_token(
    *,
    app_id: str,
    app_secret: str,
    refresh_token: str,
    timeout: int = 30,
) -> Dict[str, Any]:
    app_access_token = await get_feishu_app_access_token(
        app_id=app_id,
        app_secret=app_secret,
        timeout=timeout,
    )
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    return await _post_feishu_auth_json(
        "/authen/v1/oidc/refresh_access_token",
        payload=payload,
        timeout=timeout,
        app_access_token=app_access_token,
    )


async def get_feishu_app_access_token(
    *,
    app_id: str,
    app_secret: str,
    timeout: int = 30,
) -> str:
    response = await http_client.post(
        "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
        json={
            "app_id": app_id,
            "app_secret": app_secret,
        },
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=timeout,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Non-JSON response from Feishu app_access_token API (HTTP {response.status_code}): "
            f"{response.text[:500]}"
        ) from exc

    code = data.get("code")
    if code not in (None, 0):
        raise FeishuAuthError(int(code), str(data.get("msg") or "Unknown error"), data)
    if response.status_code >= 400:
        raise FeishuAuthError(response.status_code, str(data.get("msg") or response.reason), data)

    app_access_token = data.get("app_access_token")
    if not app_access_token:
        raise RuntimeError("Feishu app_access_token API returned no app_access_token")
    return str(app_access_token)


async def _post_feishu_auth_json(
    path: str,
    *,
    payload: Dict[str, Any],
    timeout: int,
    app_access_token: str,
) -> Dict[str, Any]:
    url = f"{FEISHU_OPEN_API_BASE_URL}{path}"
    response = await http_client.post(
        url,
        json=payload,
        headers={
            "Authorization": f"Bearer {app_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=timeout,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Non-JSON response from Feishu auth API (HTTP {response.status_code}): "
            f"{response.text[:500]}"
        ) from exc

    code = data.get("code")
    if code not in (None, 0):
        raise FeishuAuthError(int(code), str(data.get("msg") or "Unknown error"), data)
    if response.status_code >= 400:
        raise FeishuAuthError(response.status_code, str(data.get("msg") or response.reason), data)

    return data.get("data") or data


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }
    # apply to all operations so the Authorize button is available globally
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in path_item.values():
            operation.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    # Skip authentication for health/docs/openapi endpoints
    if request.url.path in (
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/sso/login",
        "/sso/callback",
        "/sso/exchange",
        "/sso/refresh",
        "/feishu-login",
        "/feishu-exchange-code",
        "/feishu-refresh-token",
    ):
        return await call_next(request)

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse(
            {"detail": "Missing authorization header"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


    try:
        resp = await http_client.get(AUTH_SERVICE_URL, headers={"Authorization": auth_header}, timeout=5.0)
    except httpx.RequestError:
        return JSONResponse(
            {"detail": "Auth service unavailable"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    if resp.status_code != 200:
        return JSONResponse(
            {"detail": "Invalid token"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    data = resp.json()
    logger.info(f"Auth service response data: {data}")
    if not data.get("success"):
        return JSONResponse(
            {"detail": "Authentication failed"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    request.state.user = data.get("data", {}).get("email", {})
    request.state.user_data = data.get("data", {})
    return await call_next(request)


def require_ou_name(required_ou: str):
    """
    Dependency factory to check if user's ou_name matches the required value.
    Usage: @app.post("/endpoint", dependencies=[Depends(require_ou_name("代号W"))])
    """
    def check_ou(request: Request):
        if not hasattr(request.state, "user_data"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User data not available"
            )
        
        user_data = request.state.user_data
        ou_name = user_data.get("ou_name", "")
        
        if ou_name != required_ou:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required organization: {required_ou}"
            )
        
        return True
    
    return check_ou


@app.get("/sso/login", response_class=HTMLResponse)
async def sso_login(request: Request):
    """
    Renders a simple HTML page with a button that redirects the user to the SSO authorize URL.
    The SSO provider should redirect back to /sso/callback?code=...
    """
    redirect_uri = request.url_for("sso_callback")
    # build authorize URL (minimal params)
    sso_url = (
        f"{SSO_AUTHORIZE_URL}"
        f"?response_type=code&scope=read"
        f"&client_id={quote(SSO_CLIENT_ID)}"
        f"&redirect_uri={quote(str(redirect_uri), safe='')}"
        f"&state=0dccb2702c5d13d213e4fd43f2fecb196WIV4rESpnq_idp"
    )

    html = f"""
    <html>
      <head><title>SSO Login</title></head>
      <body>
        <h1>Sign in with SSO</h1>
        <p><a id="sso-link" href="{sso_url}"><button>Authenticate via SSO</button></a></p>
        <p>After approving, you will be redirected back here with a <code>code</code> query parameter.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/feishu-login", response_model=FeishuLoginResponse)
def feishu_login(
    redirect_uri: str,
    state: Optional[str] = None,
    app_id: Optional[str] = None,
) -> FeishuLoginResponse:
    """Return a Feishu login URL users can open in a browser."""
    try:
        login_url = build_feishu_login_url(_require_feishu_app_id(app_id), redirect_uri, state=state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FeishuLoginResponse(login_url=login_url)


@app.post("/feishu-exchange-code")
async def feishu_exchange_code(request: FeishuExchangeCodeRequest) -> Dict[str, Any]:
    """Exchange Feishu login code for user_access_token."""
    try:
        return await exchange_feishu_code_for_user_token(
            app_id=_require_feishu_app_id(request.app_id),
            app_secret=_require_feishu_app_secret(request.app_secret),
            code=request.code,
            redirect_uri=request.redirect_uri,
            timeout=request.timeout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FeishuAuthError as exc:
        raise HTTPException(status_code=400, detail=exc.response or {"msg": exc.msg}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/feishu-refresh-token")
async def feishu_refresh_token(request: FeishuRefreshTokenRequest) -> Dict[str, Any]:
    """Refresh Feishu user_access_token with refresh_token."""
    try:
        return await refresh_feishu_user_token(
            app_id=_require_feishu_app_id(request.app_id),
            app_secret=_require_feishu_app_secret(request.app_secret),
            refresh_token=request.refresh_token,
            timeout=request.timeout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FeishuAuthError as exc:
        raise HTTPException(status_code=400, detail=exc.response or {"msg": exc.msg}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/sso/callback", response_class=HTMLResponse, name="sso_callback")
async def sso_callback(request: Request, code: str = None):
    """
    Callback endpoint for the SSO provider. Exchanges the authorization code for an access token
    and renders the token inside a textarea.
    """
    if not code:
        return HTMLResponse(content="<html><body><h2>Missing code in query</h2></body></html>", status_code=400)

    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": str(request.url.replace(query="")),  # full redirect_uri expected by provider
        "client_id": SSO_CLIENT_ID,
        "client_secret": SSO_CLIENT_SECRET,
    }

    try:
        resp = await http_client.post(SSO_TOKEN_URL, data=token_payload, timeout=10.0)
    except httpx.RequestError:
        return HTMLResponse(content="<html><body><h2>Token endpoint unavailable</h2></body></html>", status_code=503)

    if resp.status_code != 200:
        body = resp.text
        return HTMLResponse(content=f"<html><body><h2>Token exchange failed</h2><pre>{body}</pre></body></html>", status_code=resp.status_code)

    token_data = resp.json()
    access_token = token_data.get("access_token", "")

    html = f"""
    <html>
      <head><title>SSO Token</title></head>
      <body>
        <h1>Access Token</h1>
        <p>Below is the access token returned by the SSO token endpoint:</p>
        <textarea cols="100" rows="8" readonly>{access_token}</textarea>
        <h2>Full token response</h2>
        <pre>{token_data}</pre>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/sso/exchange")
async def sso_exchange(request: TokenExchangeRequest):
    """
    Exchanges an authorization code for an access token.
    Intended for frontend clients.
    """
    token_payload = {
        "grant_type": "authorization_code",
        "code": request.code,
        #not a required field, user will not be redirect to any page. only token will be returned. only provide here to maintain the schema intergrity
        "redirect_uri": request.redirect_uri,
        "client_id": SSO_CLIENT_ID,
        "client_secret": SSO_CLIENT_SECRET,
    }

    try:
        resp = await http_client.post(SSO_TOKEN_URL, data=token_payload, timeout=10.0)
    except httpx.RequestError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Token endpoint unavailable")

    if resp.status_code != 200:
        logger.error(f"Token exchange failed: {resp.text}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token exchange failed")

    return resp.json()


@app.post("/sso/refresh")
async def sso_refresh(request: TokenRefreshRequest):
    """
    Refreshes an access token using a refresh token.
    """
    token_payload = {
        "grant_type": "refresh_token",
        "refresh_token": request.refresh_token,
        "client_id": SSO_CLIENT_ID,
    }

    try:
        resp = await http_client.post(SSO_TOKEN_URL, data=token_payload, timeout=10.0)
    except httpx.RequestError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Token endpoint unavailable")

    if resp.status_code != 200:
        logger.error(f"Token refresh failed: {resp.text}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token refresh failed")

    return resp.json()


@app.get("/user")
async def get_user_info(request: Request):
    """
    Returns the full user data set by the auth middleware.
    """
    if not hasattr(request.state, "user_data"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User data not available")
    return request.state.user_data


@app.get("/health")
async def health():
    """
    Checks the health of the service, including database and S3 connectivity.
    """
    # Check Postgres connection
    postgres_status = "ok"
    try:
        async with db_pool.connection() as conn:
            await conn.execute("SELECT 1")
    except Exception as e:
        logger.error(f"Postgres health check failed: {e}")
        postgres_status = "error"

    # Check S3 connection
    s3_status = "ok"
    try:
        # boto3 is not async, run in thread to not block event loop
        
        await asyncio.to_thread(s3_client.list_buckets)
    except (NoCredentialsError, PartialCredentialsError):
        s3_status = "error: credentials not configured"
        logger.error("S3 health check failed: credentials not configured.")
    except Exception as e:
        logger.error(f"S3 health check failed: {e}")
        s3_status = "error"

    status_code = 200 if postgres_status == "ok" and s3_status == "ok" else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "postgres_status": postgres_status,
            "s3_status": s3_status,
        },
    )



async def prompt_enhance(system_promt:dict, user_prompt:dict, model:str='gemini-3.1-pro-preview'):
    response = await client_chat.chat.completions.create(
        model=model,
        messages=[
            system_promt,
            user_prompt
        ]
    )
    enhanced_prompt = response.choices[0].message.content
    return enhanced_prompt


async def generate_gemini_image_with_fallback(contents, config, model: str = "gemini-3-pro-image-preview"):
    if client is None:
        raise RuntimeError("Primary GenAI client is not initialized")
    try:
        return await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as primary_error:
        if client_backup is None:
            raise
        logger.warning(f"Primary GenAI request failed; retrying with backup client. Error: {primary_error}")
        return await client_backup.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )


def _strip_data_uri_prefix(image_base64: str) -> str:
    if "," in image_base64:
        return image_base64.split(",")[1]
    return image_base64


def _decode_base64_to_image(image_base64: str) -> Image.Image:
    raw_base64 = _strip_data_uri_prefix(image_base64)
    image_data = base64.b64decode(raw_base64)
    return Image.open(BytesIO(image_data))


def _encode_image_to_base64_png(image: Image.Image) -> str:
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _compute_canvas_size(image_width: int, image_height: int, aspect_ratio: float = 2.0) -> tuple[int, int]:
    if image_width / image_height >= aspect_ratio:
        canvas_width = image_width
        canvas_height = math.ceil(image_width / aspect_ratio)
    else:
        canvas_height = image_height
        canvas_width = math.ceil(aspect_ratio * image_height)
    return canvas_width, canvas_height


def _place_on_white_canvas_2_to_1(image: Image.Image, max_width: int = 1600) -> Image.Image:
    img = image.convert("RGBA")
    img_width, img_height = img.size
    canvas_width, canvas_height = _compute_canvas_size(img_width, img_height, aspect_ratio=2.0)

    scale = 1.0
    if max_width and canvas_width > max_width:
        scale = max_width / canvas_width

    scaled_canvas_width = max(1, int(round(canvas_width * scale)))
    scaled_canvas_height = max(1, int(round(canvas_height * scale)))
    scaled_img_width = max(1, int(round(img_width * scale)))
    scaled_img_height = max(1, int(round(img_height * scale)))

    resized_image = img.resize((scaled_img_width, scaled_img_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (scaled_canvas_width, scaled_canvas_height), (255, 255, 255))
    offset = ((scaled_canvas_width - scaled_img_width) // 2, (scaled_canvas_height - scaled_img_height) // 2)
    canvas.paste(resized_image, offset, resized_image)
    return canvas


async def _preprocess_spine_image(image_base64: str) -> str:
    source_image = _decode_base64_to_image(image_base64)
    contents = ["change backgroud to white", source_image]
    response = await generate_gemini_image_with_fallback(
        contents=contents,
        config=GenerateContentConfig(
            response_modalities=[Modality.IMAGE]
        ),
    )
    
    processed_data = None
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data:
                processed_data = part.inline_data.data
                break
        if processed_data:
            break
    
    if not processed_data:
        raise RuntimeError("Gemini preprocessing did not return image data")
    
    processed_image = Image.open(BytesIO(processed_data))
    processed_image = source_image
    processed_image = _place_on_white_canvas_2_to_1(processed_image, max_width=1600)
    processed_image = processed_image.convert("RGBA")
    return _encode_image_to_base64_png(processed_image)


@app.post("/nanobanana")
async def nanobanana(request: Request, prompt: ImageGenerationInput)-> ImageGenerationResult:
    if s3_client is None:
        raise HTTPException(status_code=503, detail="S3 service not available")

    # Check if user is allowed to generate multiple images
    if prompt.num_images > 1:
        user_email = request.state.user if hasattr(request.state, "user") else None
        if not user_email:
            raise HTTPException(status_code=403, detail="User email not available")
        
        # Parse whitelist from environment variable
        whitelist = [email.strip() for email in MULTI_IMAGE_WHITELIST.split(",") if email.strip()]
        
        if user_email not in whitelist:
            raise HTTPException(
                status_code=403, 
                detail=f"请联系技术中心，解锁多图生成功能。"
            )

    telemetry = Telemetry(db_pool,s3_client=s3_client)
    prompt_data = prompt.model_dump()
    if hasattr(request.state, "user") and isinstance(request.state.user, str):
        prompt_data["email"] = request.state.user
    prompt_id = await telemetry.save_prompt(prompt_data, "nanobanana_prompts")

    if len(prompt.base64_images) > 0:
        system_prompt = {
            'role': 'system',
            'content': 'You are a helpful assistant that can analyze images and improve the user prompts for image edit models (gemini-3-pro-image-preview). Please use the best of your abilities to understand the content of the image and user intention to provide the most accurate prompt. Always use English prompt, but keep any text that should appear in the image in its original language. Please return the enhanced prompt only.'
        }
    else:
        system_prompt = {
            'role': 'system',
            'content': 'You are a helpful assistant that can improve the user prompts for image generation models (gemini-3-pro-image-preview). Please use the best of your abilities to understand user intention to provide the most accurate and detailed prompt. Always use English prompt, but keep any text that should appear in the image in its original language. Please return the enhanced prompt only.'
        }
    if prompt.prompt_enhance:
        content = []
        for base64_image in prompt.base64_images:
            content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': base64_image
                    }
                }
            )
        content.append(
            {'type': 'text', 'text': prompt.user_prompt}
        )

        user_prompt = {
            'role': 'user',
            'content': content
        }

        

        try:
            enhanced_prompt = await prompt_enhance(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            raise HTTPException(status_code=500, detail="Failed to enhance prompt")
    else:
        enhanced_prompt = prompt.user_prompt
    
    # Prepare contents for image generation
    contents = [enhanced_prompt]
    for base64_image in prompt.base64_images:
        if base64_image.startswith("data:image/") and ";base64," in base64_image:
            base64_image = base64_image.split(",")[1]
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        contents.append(image)

    # Function to generate a single image
    async def generate_single_image():
        try:
            response = await generate_gemini_image_with_fallback(
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=[Modality.IMAGE]
                ),
            )
            
            base64_image = ''
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.inline_data:
                        base64_string = base64.b64encode(part.inline_data.data).decode('utf-8')
                        base64_image = base64_string
                    elif part.text:
                        logger.info(part.text)
            
            if not base64_image:
                return None
            return f'data:image/png;base64,{base64_image}'
        except Exception as e:
            logger.error(f"Error generating single image: {e}")
            return None

    # Generate multiple images concurrently
    try:
        tasks = [generate_single_image() for _ in range(prompt.num_images)]
        generated_images = await asyncio.gather(*tasks)
        
        # Filter out None values (failed generations)
        successful_images = [img for img in generated_images if img is not None]
        
        if not successful_images:
            raise HTTPException(status_code=500, detail="All image generation attempts failed")
        
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate images")
    
    # Save each generated image as a separate row in the database concurrently
    save_tasks = [
        telemetry.save_result({
            'base64_image': base64_image,
            'prompt_id': prompt_id
        }, "nanobanana_results")
        for base64_image in successful_images
    ]
    await asyncio.gather(*save_tasks)
    
    result = ImageGenerationResult(base64_images=successful_images)
    return result


@app.post("/sagemaker/batch")
async def sagemaker_batch(req: Request, request: BatchJobInput) -> BatchJobStatus:
    endpoint_name = "qwen-image-w-lora"
    
    # Check backlog
    try:
        backlog = await asyncio.to_thread(check_backlog, endpoint_name)
    except Exception as e:
        logger.error(f"Failed to check backlog: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check SageMaker backlog: {str(e)}")

    if backlog > 0:
        return BatchJobStatus(status="busy", message=f"SageMaker is currently processing {backlog} jobs. Please try again later.")
    
    # Generate prompts
    try:
        if request.lora_name == "icons":
            prompts = await asyncio.to_thread(generate_prompts_for_items, request.keywords, prompts_per_item=1, system_prompt=ICON_SYSTEM_PROMPT)
        else:
            prompts = await asyncio.to_thread(generate_prompts_for_items, request.keywords, prompts_per_item=4)
    except Exception as e:
        logger.error(f"Failed to generate prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate prompts: {str(e)}")

    # Submit jobs
    try:
       
        result = await asyncio.to_thread(submit_batch_jobs, prompts, endpoint_name=endpoint_name, lora_name=request.lora_name)
        count = result["count"]
        inference_ids = result["inference_ids"]
        output_paths = result["output_paths"]
    except Exception as e:
        logger.error(f"Failed to submit batch jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit batch jobs: {str(e)}")
    
    # Track jobs in database using Telemetry (best-effort, non-blocking)
    if s3_client_us_east_2 and db_pool:
        telemetry = Telemetry(db_pool, s3_client=s3_client_us_east_2)
        user_email = req.state.user if hasattr(req.state, "user") else "unknown"
        
        for i, (inference_id, output_path) in enumerate(zip(inference_ids, output_paths)):
            try:
                job_data = {
                    "email": user_email,
                    "space": request.space,
                    "keywords": request.keywords,
                    "output_path": output_path
                }
                await telemetry.save_prompt(job_data, "sagemaker_jobs")
            except Exception as e:
                logger.warning(f"Failed to track job {inference_id} in database: {e}")
        
    return BatchJobStatus(status="success", message="Batch jobs submitted successfully", job_count=count)


@app.get("/sagemaker/output")
async def list_sagemaker_output(req: Request, limit: int = 10, space: str = None):
    # Validate limit
    if not isinstance(limit, int) or limit < 1 or limit > 300:
        raise HTTPException(status_code=400, detail="limit must be an integer between 1 and 300")
    
    # Validate space
    if space and space not in ["public", "private"]:
        raise HTTPException(status_code=400, detail="space must be either 'public' or 'private'")
    
    region = "us-east-2"
    prefix = "diffusers-async/output/"
    user_email = req.state.user if hasattr(req.state, "user") else None
    
    # Get user's output paths from database (best-effort)
    user_output_paths = set()
    if db_pool:
        try:
            async with db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    if space == "public":
                        # Public space: select all public records
                        await cur.execute(
                            "SELECT output_path FROM sagemaker_jobs WHERE space = %s",
                            ("public",)
                        )
                    elif space == "private":
                        # Private space: only user's own private records
                        if not user_email:
                            raise HTTPException(status_code=401, detail="User email not available")
                        await cur.execute(
                            "SELECT output_path FROM sagemaker_jobs WHERE space = %s AND email = %s",
                            ("private", user_email)
                        )
                    else:
                        # No space filter: user's own records (all spaces)
                        if not user_email:
                            raise HTTPException(status_code=401, detail="User email not available")
                        await cur.execute(
                            "SELECT output_path FROM sagemaker_jobs WHERE email = %s",
                            (user_email,)
                        )
                    rows = await cur.fetchall()
                    user_output_paths = {row[0] for row in rows if row[0]}
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to query user's output paths from database: {e}")
    
    try:
        def get_sagemaker_files():
            boto_session = boto3.Session(region_name=region)
            sess = sagemaker.Session(boto_session=boto_session)
            bucket_name = sess.default_bucket()
            s3 = boto_session.client("s3")
            return s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        response = await asyncio.to_thread(get_sagemaker_files)
        
        files = []
        if not user_output_paths:
            return files # no files for user
        if 'Contents' in response:
            # Sort by LastModified desc to get latest
            sorted_contents = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            for obj in sorted_contents:
                
                if user_output_paths:
                    s3_uri = f"s3://{response.get('Name', '')}/{obj['Key']}"
                    if s3_uri not in user_output_paths:
                        continue
               
                
                files.append({
                    "key": obj['Key'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "size": obj['Size']
                })
                
                # Apply limit after filtering
                if len(files) >= limit:
                    break
            
        return files
            
    except Exception as e:
        logger.error(f"Failed to list S3 objects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list S3 objects: {str(e)}")


@app.post("/seedream")
async def seedream(request: Request, prompt: ImageGenerationInput) -> ImageGenerationResult:
    if s3_client is None:
        raise HTTPException(status_code=503, detail="S3 service not available")
    
    telemetry = Telemetry(db_pool, s3_client=s3_client)
    prompt_data = {
        "user_prompt": prompt.user_prompt,
        "base64_images": prompt.base64_images,
        "prompt_enhance": prompt.prompt_enhance
    }
    if hasattr(request.state, "user") and isinstance(request.state.user, str):
        prompt_data["email"] = request.state.user
    prompt_id = await telemetry.save_prompt(prompt_data, "seedream_prompts")

    enhanced_prompt = prompt.user_prompt
    if prompt.prompt_enhance:
        system_prompt = {
            'role': 'system',
            'content': SEEDREAM_SYTEM_PROMPT
        }
        
        content = []
        for base64_image in prompt.base64_images:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': base64_image
                }
            })
            
        content.append({
            'type': 'text', 
            'text': prompt.user_prompt
        })

        user_prompt = {
            'role': 'user',
            'content': content
        }
        try:
            enhanced_prompt = await prompt_enhance(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            raise HTTPException(status_code=500, detail="Failed to enhance prompt")

    image_urls = []
    if prompt.base64_images:
        try:
            bucket_name = os.getenv("S3_BUCKET", "seedream")
            
            def upload_and_sign(b64_img):
                if "," in b64_img:
                    base64_data = b64_img.split(",")[1]
                else:
                    base64_data = b64_img
                
                image_data = base64.b64decode(base64_data)
                
                # Resize and compress image
                with Image.open(BytesIO(image_data)) as img:
                    if img.width > 1600:
                        new_height = int(img.height * (1600 / img.width))
                        img = img.resize((1600, new_height), Image.Resampling.LANCZOS)
                    
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    output_buffer = BytesIO()
                    img.save(output_buffer, format='JPEG', quality=90)
                    image_data = output_buffer.getvalue()

                file_key = f"seedream/{uuid.uuid4()}.jpg"
                
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=file_key,
                    Body=image_data,
                    ContentType='image/jpeg'
                )
                url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': file_key},
                    ExpiresIn=3600
                )
                logger.info(f"Uploaded image to S3: {url}")
                return url

            tasks = [asyncio.to_thread(upload_and_sign, img) for img in prompt.base64_images]
            image_urls = list(await asyncio.gather(*tasks))
        except Exception as e:
             logger.error(f"S3 upload error: {e}")
             raise HTTPException(status_code=500, detail=str(e))

    try:
        # Call OpenAI API
        kwargs = {
            "model": "seedream-4.5",
            "prompt": enhanced_prompt,
        }
        if image_urls:
            kwargs["extra_body"] = {"image": image_urls,"size":"2k","watermark ":False}

        result = await client_seedream.images.generate(**kwargs)
        
        base64_image = ""
        if result.data and len(result.data) > 0:
            if result.data[0].url:
                try:
                    img_resp = await http_client.get(result.data[0].url)
                    if img_resp.status_code == 200:
                        base64_image = base64.b64encode(img_resp.content).decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to download image: {e}")

        if base64_image:
             result = ImageGenerationResult(base64_images=[f"data:image/png;base64,{base64_image}"])
             result_json = result.model_dump()
             result_json['prompt_id'] = prompt_id
             result_json.pop('base64_images', None)  # remove base64_images before saving
             result_json["base64_image"] = f"data:image/png;base64,{base64_image}"  # save single image base64 for telemetry
             
             await telemetry.save_result(result_json, "seedream_results")
             
             return result
        
        raise HTTPException(status_code=500, detail="Failed to generate or retrieve image")

    except Exception as e:
        logger.error(f"Seedream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/w-edit")
async def w_edit(request: Request, input_data: WEditInput) -> WEditResult:
    """
    Synchronous SageMaker image editing endpoint for W character editing.
    Sends a request to SageMaker endpoint and waits for the result.
    Only one request can be processed at a time.
    """
    # Check if lock is available, return error if already locked
    # Be Aware(SUPER IMPORTANT), the lock will not work across multiple uvicorn processes. For Scaling, use Distributed Locking Mechanism like Redis Lock.
    if w_edit_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Another image editing request is currently being processed. Please try again in 45 seconds."
        )
    
    await w_edit_lock.acquire()
    try:
        endpoint_name = "qwen-edit"
        region = os.getenv("AWS_REGION", "us-west-2")
        
        # Extract base64 image data (without prefix) for SageMaker
        image_base64 = input_data.image
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        logger.info(f"Received W edit request with lora_name: {input_data.lora_name}")

        if input_data.lora_name == "spine":
            try:
                logger.info("Preprocessing spine image with Gemini for white background")
                image_base64 = await _preprocess_spine_image(input_data.image)
            except Exception as e:
                logger.error(f"Spine image preprocessing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Spine preprocessing failed: {str(e)}")
        
        # Generate prompts using LLM only if lora_name is "char"
        if input_data.lora_name == "char":
            # Ensure base64 image has data URI prefix for prompt_enhance
            image_with_prefix = input_data.image
            if not image_with_prefix.startswith("data:image/"):
                image_with_prefix = f"data:image/png;base64,{input_data.image}"
            
            system_prompt = {
                'role': 'system',
                'content': WCharEdit_SYSTEM_PROMPT
            }
            
            user_prompt = {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_with_prefix
                        }
                    }
                ]
            }
            
            try:
                enhanced_response = await prompt_enhance(system_prompt, user_prompt,model="gpt-5.2")
                # Parse JSON response
                prompt_data = json.loads(enhanced_response)
                positive_prompt = prompt_data.get("positive_prompt", "")
                negative_prompt = prompt_data.get("negative_prompt", "")
            except Exception as e:
                logger.error(f"Error generating prompts with LLM: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate prompts: {str(e)}")
        elif input_data.lora_name == "spine":
            positive_prompt = WSpine_EDIT_POSITIVE_PROMPT
            negative_prompt = ""
        elif input_data.lora_name == "vic-item-icon":
            logger.info("Using vic-item-icon prompts with item lora behavior")
            positive_prompt = VICItem_EDIT_POSITIVE_PROMPT
            negative_prompt = VICItem_EDIT_NEGATIVE_PROMPT
        else:
            logger.info("Skipping LLM prompt generation for item editing")
            positive_prompt = WItem_EDIT_POSITIVE_PROMPT
            negative_prompt = WItem_EDIT_NEGATIVE_PROMPT
        payload = {
            "image": image_base64,
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": input_data.num_inference_steps,
            "true_cfg_scale": input_data.true_cfg_scale,
            "guidance_scale": input_data.guidance_scale,
            "num_images_per_prompt": input_data.num_images_per_prompt,
            "lora_name": input_data.lora_name,
        }
        
        # Add seed if specified
        if input_data.seed != -1:
            payload["seed"] = input_data.seed
        
        sagemaker_failed = False
        try:
            # Initialize SageMaker async predictor in a thread
            def make_async_prediction():
                boto_session = boto3.Session(region_name=region)
                sess = sagemaker.Session(boto_session=boto_session)
                base_predictor = sagemaker.predictor.Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=sess
                )
                predictor = AsyncPredictor(predictor=base_predictor, name=endpoint_name)
                predictor.serializer = JSONSerializer()
                #predictor.deserializer = JSONDeserializer()
                
                
                # Invoke async endpoint and wait for result
                response = predictor.predict_async(payload)
                return response.get_result(waiter_config=sagemaker.async_inference.waiter_config.WaiterConfig(
                            max_attempts=60,  # Wait up to 10 minutes (60 * 10 seconds)
                            delay=10
                        ))
            
            logger.info(f"Sending async request to SageMaker endpoint: {endpoint_name}")
            result = await asyncio.to_thread(make_async_prediction)
        except Exception as e:
            logger.error(f"SageMaker async request failed: {e}")
            sagemaker_failed = True
            
            # Try Vertex AI as backup
            if VERTEX_PROJECT and VERTEX_ENDPOINT_ID:
                logger.info("Falling back to Vertex AI endpoint...")
                try:
                    from google.cloud import aiplatform
                    
                    def call_vertex_endpoint():
                        # Initialize Vertex AI
                        aiplatform.init(project=VERTEX_PROJECT, location=VERTEX_REGION)
                        
                        # Get endpoint
                        if VERTEX_ENDPOINT_ID.startswith("projects/"):
                            endpoint = aiplatform.Endpoint(VERTEX_ENDPOINT_ID)
                        else:
                            endpoint = aiplatform.Endpoint(
                                endpoint_name=f"projects/{VERTEX_PROJECT}/locations/{VERTEX_REGION}/endpoints/{VERTEX_ENDPOINT_ID}"
                            )
                        
                        # Prepare request for Vertex AI (use original payload)
                        request_body = {
                            "image": image_base64,
                            "prompt": positive_prompt,
                            "negative_prompt": negative_prompt,
                            "num_inference_steps": input_data.num_inference_steps,
                            "true_cfg_scale": input_data.true_cfg_scale,
                            "guidance_scale": input_data.guidance_scale,
                            "num_images_per_prompt": input_data.num_images_per_prompt,
                            "lora_name": input_data.lora_name,
                        }
                        
                        if input_data.seed != -1:
                            request_body["seed"] = input_data.seed
                        
                        # Make prediction
                        response = endpoint.predict(instances=[request_body])
                        
                        # Parse response
                        predictions = response.predictions
                        if not predictions:
                            raise ValueError("No predictions returned from Vertex AI endpoint")
                        
                        return predictions[0]
                    
                    logger.info(f"Calling Vertex AI endpoint: {VERTEX_ENDPOINT_ID}")
                    result = await asyncio.to_thread(call_vertex_endpoint)
                    logger.info("Successfully received response from Vertex AI")
                    
                except Exception as vertex_error:
                    logger.error(f"Vertex AI backup also failed: {vertex_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"SageMaker failed: {str(e)}. Vertex AI backup also failed: {str(vertex_error)}"
                    )
            else:
                logger.warning("Vertex AI backup not configured (missing VERTEX_PROJECT or VERTEX_ENDPOINT_ID)")
                raise HTTPException(
                    status_code=500, 
                    detail=f"SageMaker async request failed: {str(e)}"
                )
    finally:
        if w_edit_lock.locked():
            w_edit_lock.release()
            
    # Post-processing (outside lock)
    try:
        # Handle list response (SageMaker sometimes wraps response in a list)


        if isinstance(result, bytes):
            result = json.loads(result.decode("utf-8"))
            print("Decoded bytes -> JSON dict")
        elif isinstance(result, str):
            result = json.loads(result)
            print("Parsed string -> JSON dict")


        if isinstance(result, list) and len(result) > 0:
            result = json.loads(result[0]) if isinstance(result[0], str) else result[0]
        
        # Extract images
        if isinstance(result, dict) and "images_base64" in result:
            images = result["images_base64"]
            meta = result.get("meta", {})
            
            logger.info(f"Successfully received {len(images)} image(s) from SageMaker")
            
            # Post-process images with nanobanana for outline enhancement (only for char lora)
            enhanced_images = []
            if input_data.lora_name == "char":
                for img_base64 in images:
                    try:
                        '''
                        if not img_base64.startswith("data:image/"):
                            img_data_uri = f"data:image/png;base64,{img_base64}"
                        else:
                            img_data_uri = img_base64
                        
                        
                        if "," in img_data_uri:
                            base64_data = img_data_uri.split(",")[1]
                        else:
                            base64_data = img_data_uri
                        '''
                        image_data = base64.b64decode(img_base64)
                        image = Image.open(BytesIO(image_data))
                        
                        # Use nanobanana for outline enhancement
                        contents = [WCharEdit_OUTLINE_ENHANCE_SYSTEM_PROMPT, image]
                        
                        response = await generate_gemini_image_with_fallback(
                            contents=contents,
                            config=GenerateContentConfig(
                                response_modalities=[Modality.IMAGE]
                            ),
                        )
                        
                        enhanced_image = ''
                        for candidate in response.candidates:
                            for part in candidate.content.parts:
                                if part.inline_data:
                                    enhanced_image = base64.b64encode(part.inline_data.data).decode('utf-8')
                                    break
                                elif part.text:
                                    logger.info(f"Nanobanana response text: {part.text}")
                            if enhanced_image:
                                break
                        
                        if enhanced_image:
                            enhanced_images.append(f"data:image/png;base64,{enhanced_image}")
                        else:
                            # Fallback to original if enhancement fails
                            logger.warning("Nanobanana enhancement failed, using original image")
                            if not img_base64.startswith("data:image/"):
                                enhanced_images.append(f"data:image/png;base64,{img_base64}")
                            else:
                                enhanced_images.append(img_base64)
                            
                    except Exception as e:
                        logger.error(f"Error in nanobanana post-processing: {e}")
                        # Fallback to original image on error
                        if not img_base64.startswith("data:image/"):
                            enhanced_images.append(f"data:image/png;base64,{img_base64}")
                        else:
                            enhanced_images.append(img_base64)
            else:
                # For item lora, skip post-processing and return images directly
                for img_base64 in images:
                    if not img_base64.startswith("data:image/"):
                        enhanced_images.append(f"data:image/png;base64,{img_base64}")
                    else:
                        enhanced_images.append(img_base64)
            
            return WEditResult(
                images_base64=enhanced_images,
                meta=meta
            )
        else:
            logger.error(f"Unexpected result format from SageMaker: {type(result)}")
            raise HTTPException(
                status_code=500, 
                detail="Unexpected response format from SageMaker endpoint"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Post-processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Post-processing failed: {str(e)}"
        )


@app.post("/w-t2i")
async def w_t2i(req: Request, input_data: WT2IInput) -> WT2IResult:
    """
    Synchronous SageMaker image generation endpoint for icons and props.
    Generates a prompt based on the keyword and lora_name, then calls SageMaker endpoint.
    Waits for the result and returns generated images.
    """
    # Check if lock is available, return error if already locked
    # Be Aware(SUPER IMPORTANT), the lock will not work across multiple uvicorn processes. For Scaling, use Distributed Locking Mechanism like Redis Lock.
    if w_t2i_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Another image generation request is currently being processed. Please try again in 45 seconds."
        )

    await w_t2i_lock.acquire()
    endpoint_name = "qwen-t2i"
    region = "ap-south-1"
    
    try:
        # Generate prompt using LLM based on lora_name
        try:
            if input_data.lora_name == "icons":
                prompts_dict = await asyncio.to_thread(
                    generate_prompts_for_items,
                    [input_data.keyword],
                    prompts_per_item=1,
                    system_prompt=ICON_SYSTEM_PROMPT
                )
            else:
                prompts_dict = await asyncio.to_thread(
                    generate_prompts_for_items,
                    [input_data.keyword],
                    prompts_per_item=1
                )

            # Extract the prompt from the returned dictionary
            # prompts_dict is like {"keyword": [{"positive": "...", "negative": "..."}]}
            if not prompts_dict or input_data.keyword not in prompts_dict:
                raise ValueError("Failed to generate prompt")

            prompt_list = prompts_dict[input_data.keyword]
            if not prompt_list or len(prompt_list) == 0:
                raise ValueError("No prompts generated")

            # Get the positive and negative prompts from the first result
            prompt = prompt_list[0].get("positive", "")
            negative_prompt = prompt_list[0].get("negative", "")
            if not prompt:
                raise ValueError("Empty prompt generated")

            logger.info(f"Generated prompt: {prompt}")
            logger.info(f"Generated negative prompt: {negative_prompt}")

        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate prompt: {str(e)}")

        # Build payload for SageMaker
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": input_data.num_inference_steps,
            "guidance_scale": input_data.guidance_scale,
            "num_images_per_prompt": input_data.num_images_per_prompt,
            "lora_name": input_data.lora_name
        }

        # Add seed if specified
        if input_data.seed != -1:
            payload["seed"] = input_data.seed

        try:
            # Initialize SageMaker predictor and make synchronous prediction
            def make_prediction():
                boto_session = boto3.Session(region_name=region)
                sess = sagemaker.Session(boto_session=boto_session)

                predictor = Predictor(
                    endpoint_name=endpoint_name,
                    sagemaker_session=sess,
                    serializer=JSONSerializer(),
                    deserializer=JSONDeserializer()
                )

                return predictor.predict(payload)

            logger.info(f"Sending synchronous request to SageMaker endpoint: {endpoint_name}")
            result = await asyncio.to_thread(make_prediction)

        except Exception as e:
            logger.error(f"SageMaker sync request failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"SageMaker request failed: {str(e)}"
            )

        # Parse result
        try:
            # Handle list response (SageMaker sometimes wraps response in a list)
            if isinstance(result, list) and len(result) > 0:
                result = json.loads(result[0]) if isinstance(result[0], str) else result[0]

            # Extract images
            if isinstance(result, dict) and "images_base64" in result:
                images = result["images_base64"]
                meta = result.get("meta", {})

                logger.info(f"Successfully received {len(images)} image(s) from SageMaker")

                # Add data URI prefix to images if not present
                formatted_images = []
                for img_base64 in images:
                    if not img_base64.startswith("data:image/"):
                        formatted_images.append(f"data:image/png;base64,{img_base64}")
                    else:
                        formatted_images.append(img_base64)

                return WT2IResult(
                    images_base64=formatted_images,
                    prompt=prompt,
                    meta=meta
                )
            else:
                logger.error(f"Unexpected result format from SageMaker: {type(result)}")
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected response format from SageMaker endpoint"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Result parsing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Result parsing failed: {str(e)}"
            )
    finally:
        if w_t2i_lock.locked():
            w_t2i_lock.release()





