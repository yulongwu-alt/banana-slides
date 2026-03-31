"""Authentication middleware helpers and SSO routes."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from flask import Blueprint, current_app, g, jsonify, request, url_for

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)

PUBLIC_AUTH_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/sso/login",
    "/api/sso/callback",
    "/api/sso/exchange",
    "/api/sso/refresh",
}

SSO_STATE = "0dccb2702c5d13d213e4fd43f2fecb196WIV4rESpnq_idp"


class UpstreamUnavailable(RuntimeError):
    """Raised when an upstream auth service cannot be reached."""


def _parse_json(body: str):
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _redact_token_payload(payload: dict[str, str]) -> dict[str, str]:
    """Redact sensitive token payload fields before logging."""
    redacted = dict(payload)
    for key in ("client_secret", "refresh_token", "code"):
        if redacted.get(key):
            redacted[key] = "***"
    return redacted


def _http_json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
    timeout: float = 5.0,
):
    if not url:
        raise UpstreamUnavailable("Upstream URL is not configured")

    request_headers = dict(headers or {})
    payload = None
    if form_data is not None:
        payload = urllib.parse.urlencode(form_data).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

    req = urllib.request.Request(url, data=payload, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.getcode(), body, _parse_json(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Upstream HTTP error: method=%s url=%s status=%s response_body=%s",
            method,
            url,
            exc.code,
            body,
        )
        return exc.code, body, _parse_json(body)
    except urllib.error.URLError as exc:
        raise UpstreamUnavailable(str(exc)) from exc


def fetch_auth_user_data(auth_header: str) -> dict | None:
    """Validate the bearer token and return the auth service payload."""
    status, _body, data = _http_json_request(
        current_app.config.get("AUTH_SERVICE_URL", ""),
        method="GET",
        headers={"Authorization": auth_header},
        timeout=5.0,
    )
    if status != 200:
        return None
    return data if isinstance(data, dict) else {}


def _get_request_auth_header() -> str | None:
    """Return the bearer token from the request header or auth cookie."""
    auth_header = request.headers.get("Authorization")
    if auth_header:
        return auth_header

    access_token = request.cookies.get("access_token")
    if access_token:
        return f"Bearer {access_token}"

    return None


def register_auth_middleware(app):
    """Register request authentication middleware on the Flask app."""

    @app.before_request
    def auth_middleware():
        if request.method == "OPTIONS":
            return None

        if request.path in PUBLIC_AUTH_PATHS:
            return None

        auth_header = _get_request_auth_header()
        if not auth_header:
            return jsonify({"detail": "Missing authorization token"}), 401

        try:
            data = fetch_auth_user_data(auth_header)
        except UpstreamUnavailable:
            return jsonify({"detail": "Auth service unavailable"}), 503

        if data is None:
            return jsonify({"detail": "Invalid token"}), 401

        logger.info("Auth service response data: %s", data)
        if not data.get("success"):
            return jsonify({"detail": "Authentication failed"}), 401

        g.user = data.get("data", {}).get("email", {})
        g.user_data = data.get("data", {})
        return None


@auth_bp.route("/api/sso/login", methods=["GET"])
def sso_login():
    """Render a simple login page that redirects to the configured SSO provider."""
    redirect_uri = url_for("auth.sso_callback", _external=True)
    sso_url = (
        f"{current_app.config.get('SSO_AUTHORIZE_URL', '')}"
        f"?response_type=code&scope=read"
        f"&client_id={urllib.parse.quote(current_app.config.get('SSO_CLIENT_ID', ''))}"
        f"&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
        f"&state={SSO_STATE}"
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
    return html


@auth_bp.route("/api/sso/callback", methods=["GET"], endpoint="sso_callback")
def sso_callback():
    """Exchange the returned authorization code and render the token response."""
    code = request.args.get("code")
    if not code:
        return "<html><body><h2>Missing code in query</h2></body></html>", 400

    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": request.base_url,
        "client_id": current_app.config.get("SSO_CLIENT_ID", ""),
        "client_secret": current_app.config.get("SSO_CLIENT_SECRET", ""),
    }

    try:
        status, body, data = _http_json_request(
            current_app.config.get("SSO_TOKEN_URL", ""),
            method="POST",
            form_data=token_payload,
            timeout=10.0,
        )
    except UpstreamUnavailable:
        return "<html><body><h2>Token endpoint unavailable</h2></body></html>", 503

    if status != 200:
        return f"<html><body><h2>Token exchange failed</h2><pre>{body}</pre></body></html>", status

    token_data = data if isinstance(data, dict) else {}
    access_token = token_data.get("access_token", "")
    token_pre = json.dumps(token_data, ensure_ascii=False, indent=2)
    html = f"""
    <html>
      <head><title>SSO Token</title></head>
      <body>
        <h1>Access Token</h1>
        <p>Below is the access token returned by the SSO token endpoint:</p>
        <textarea cols="100" rows="8" readonly>{access_token}</textarea>
        <h2>Full token response</h2>
        <pre>{token_pre}</pre>
      </body>
    </html>
    """
    return html


@auth_bp.route("/api/sso/exchange", methods=["POST"])
def sso_exchange():
    """Exchange an authorization code for an access token."""
    payload = request.get_json(silent=True) or {}
    code = payload.get("code")
    redirect_uri = payload.get("redirect_uri")
    if not code or not redirect_uri:
        return jsonify({"detail": "code and redirect_uri are required"}), 400

    token_payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": current_app.config.get("SSO_CLIENT_ID", ""),
        "client_secret": current_app.config.get("SSO_CLIENT_SECRET", ""),
    }

    logger.info(
        "SSO exchange request: method=%s path=%s redirect_uri=%s upstream=%s payload=%s",
        request.method,
        request.path,
        redirect_uri,
        current_app.config.get("SSO_TOKEN_URL", ""),
        _redact_token_payload(token_payload),
    )

    try:
        status, body, data = _http_json_request(
            current_app.config.get("SSO_TOKEN_URL", ""),
            method="POST",
            form_data=token_payload,
            timeout=10.0,
        )
    except UpstreamUnavailable:
        return jsonify({"detail": "Token endpoint unavailable"}), 503

    if status != 200:
        logger.error(
            "Token exchange failed: upstream=%s status=%s response_body=%s",
            current_app.config.get("SSO_TOKEN_URL", ""),
            status,
            body,
        )
        return jsonify({"detail": "Token exchange failed"}), 401

    return jsonify(data if isinstance(data, dict) else {})


@auth_bp.route("/api/sso/exchange", methods=["GET", "PUT", "PATCH", "DELETE"])
def sso_exchange_method_not_allowed():
    """Log unsupported methods hitting the exchange endpoint."""
    logger.error(
        "Unsupported method for /api/sso/exchange: method=%s path=%s query=%s",
        request.method,
        request.path,
        request.query_string.decode("utf-8", errors="replace"),
    )
    return jsonify({"detail": "Method not allowed"}), 405


@auth_bp.route("/api/sso/refresh", methods=["POST"])
def sso_refresh():
    """Refresh an access token using a refresh token."""
    payload = request.get_json(silent=True) or {}
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        return jsonify({"detail": "refresh_token is required"}), 400

    token_payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": current_app.config.get("SSO_CLIENT_ID", ""),
    }

    try:
        status, body, data = _http_json_request(
            current_app.config.get("SSO_TOKEN_URL", ""),
            method="POST",
            form_data=token_payload,
            timeout=10.0,
        )
    except UpstreamUnavailable:
        return jsonify({"detail": "Token endpoint unavailable"}), 503

    if status != 200:
        logger.error("Token refresh failed: %s", body)
        return jsonify({"detail": "Token refresh failed"}), 401

    return jsonify(data if isinstance(data, dict) else {})
