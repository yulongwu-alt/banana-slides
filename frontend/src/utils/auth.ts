const ACCESS_TOKEN_COOKIE = 'access_token';
const REFRESH_TOKEN_COOKIE = 'refresh_token';
const ACCESS_TOKEN_TTL_SECONDS = 60 * 60;
const REFRESH_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30;
const SSO_REDIRECT_URI = 'https://slides.yo-star.com';

export const SSO_AUTHORIZE_URL =
  'https://idaas.yostar.net/oauth/authorize?response_type=code&scope=read&client_id=6cda083476ddfa8af792191027b982b2yq5BqELBMA8&redirect_uri=https%3A%2F%2Fslides.yo-star.com&state=75d9906504243e40b0d1f56fae03aaedqlkAcgTkpu9_idp';

type TokenResponse = {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number | string;
  refresh_expires_in?: number | string;
};

type AuthResult = 'authenticated' | 'redirected';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function getCookie(name: string): string | null {
  const prefix = `${name}=`;
  const cookies = document.cookie ? document.cookie.split('; ') : [];
  for (const cookie of cookies) {
    if (cookie.startsWith(prefix)) {
      return decodeURIComponent(cookie.slice(prefix.length));
    }
  }
  return null;
}

function setCookie(name: string, value: string, maxAgeSeconds: number) {
  const secure = window.location.protocol === 'https:' ? '; Secure' : '';
  document.cookie = `${name}=${encodeURIComponent(value)}; Path=/; Max-Age=${maxAgeSeconds}; SameSite=Lax${secure}`;
}

function clearCookie(name: string) {
  document.cookie = `${name}=; Path=/; Max-Age=0; SameSite=Lax`;
}

function parseTtl(value: number | string | undefined, fallbackSeconds: number): number {
  const parsed = typeof value === 'string' ? Number(value) : value;
  return Number.isFinite(parsed) && parsed && parsed > 0 ? parsed : fallbackSeconds;
}

async function postJson<T>(url: string, payload: Record<string, string>): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getAccessToken(): string | null {
  return getCookie(ACCESS_TOKEN_COOKIE);
}

export function getRefreshToken(): string | null {
  return getCookie(REFRESH_TOKEN_COOKIE);
}

export function clearAuthCookies() {
  clearCookie(ACCESS_TOKEN_COOKIE);
  clearCookie(REFRESH_TOKEN_COOKIE);
}

export function storeAuthTokens(tokens: TokenResponse) {
  if (tokens.access_token) {
    setCookie(
      ACCESS_TOKEN_COOKIE,
      tokens.access_token,
      parseTtl(tokens.expires_in, ACCESS_TOKEN_TTL_SECONDS),
    );
  }

  if (tokens.refresh_token) {
    setCookie(
      REFRESH_TOKEN_COOKIE,
      tokens.refresh_token,
      parseTtl(tokens.refresh_expires_in, REFRESH_TOKEN_TTL_SECONDS),
    );
  }
}

export function redirectToSso() {
  clearAuthCookies();
  window.location.assign(SSO_AUTHORIZE_URL);
}

export async function exchangeCodeForTokens(code: string) {
  const tokens = await postJson<TokenResponse>('/api/sso/exchange', {
    code,
    redirect_uri: SSO_REDIRECT_URI,
  });
  storeAuthTokens(tokens);
  return tokens;
}

export async function refreshAccessToken() {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    throw new Error('Missing refresh token');
  }

  const tokens = await postJson<TokenResponse>('/api/sso/refresh', {
    refresh_token: refreshToken,
  });

  storeAuthTokens({
    ...tokens,
    refresh_token: tokens.refresh_token ?? refreshToken,
  });

  return tokens;
}

export async function ensureAuthenticated(code: string | null): Promise<AuthResult> {
  if (code) {
    try {
      await exchangeCodeForTokens(code);
      return 'authenticated';
    } catch (error) {
      console.error('Failed to exchange SSO code', error);
      window.alert('SSO code exchange failed. Redirecting to sign in again in 2 seconds.');
      await delay(2000);
      redirectToSso();
      return 'redirected';
    }
  }

  const accessToken = getAccessToken();
  if (!accessToken) {
    redirectToSso();
    return 'redirected';
  }

  try {
    await refreshAccessToken();
    return 'authenticated';
  } catch (error) {
    console.error('Failed to refresh SSO session', error);
    redirectToSso();
    return 'redirected';
  }
}
