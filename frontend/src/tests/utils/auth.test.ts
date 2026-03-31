import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  clearAuthCookies,
  exchangeCodeForTokens,
  getAccessToken,
  getRefreshToken,
  refreshAccessToken,
  storeAuthTokens,
} from '@/utils/auth';

describe('auth utils', () => {
  beforeEach(() => {
    clearAuthCookies();
    vi.mocked(global.fetch).mockReset();
  });

  afterEach(() => {
    clearAuthCookies();
  });

  it('stores access and refresh tokens in cookies', () => {
    storeAuthTokens({
      access_token: 'access-token',
      refresh_token: 'refresh-token',
      expires_in: 120,
      refresh_expires_in: 240,
    });

    expect(getAccessToken()).toBe('access-token');
    expect(getRefreshToken()).toBe('refresh-token');
  });

  it('exchanges an auth code and persists returned tokens', async () => {
    vi.mocked(global.fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          access_token: 'new-access',
          refresh_token: 'new-refresh',
          expires_in: 300,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    const tokens = await exchangeCodeForTokens('auth-code');

    expect(tokens.access_token).toBe('new-access');
    expect(getAccessToken()).toBe('new-access');
    expect(getRefreshToken()).toBe('new-refresh');
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sso/exchange',
      expect.objectContaining({
        method: 'POST',
      }),
    );
  });

  it('refreshes the session with the refresh token cookie', async () => {
    storeAuthTokens({
      access_token: 'stale-access',
      refresh_token: 'saved-refresh',
    });

    vi.mocked(global.fetch).mockResolvedValue(
      new Response(
        JSON.stringify({
          access_token: 'fresh-access',
          expires_in: 600,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );

    await refreshAccessToken();

    expect(getAccessToken()).toBe('fresh-access');
    expect(getRefreshToken()).toBe('saved-refresh');
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sso/refresh',
      expect.objectContaining({
        method: 'POST',
      }),
    );
  });
});
