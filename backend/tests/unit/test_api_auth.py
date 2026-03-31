"""Authentication and SSO endpoint tests."""

from unittest.mock import patch

from controllers.auth_controller import UpstreamUnavailable


class TestAuthMiddleware:
    def test_protected_route_requires_authorization_token(self, unauthenticated_client):
        response = unauthenticated_client.get('/api/projects')

        assert response.status_code == 401
        assert response.get_json() == {'detail': 'Missing authorization token'}

    def test_protected_route_accepts_access_token_cookie(self, unauthenticated_client):
        unauthenticated_client.set_cookie('access_token', 'cookie-token')

        with patch(
            'controllers.auth_controller.fetch_auth_user_data',
            return_value={'success': True, 'data': {'email': 'cookie@example.com'}},
        ) as fetch_auth_user_data:
            response = unauthenticated_client.get('/api/projects')

        assert response.status_code == 200
        fetch_auth_user_data.assert_called_once_with('Bearer cookie-token')

    def test_protected_route_rejects_invalid_token(self, unauthenticated_client):
        with patch('controllers.auth_controller.fetch_auth_user_data', return_value=None):
            response = unauthenticated_client.get(
                '/api/projects',
                headers={'Authorization': 'Bearer bad-token'},
            )

        assert response.status_code == 401
        assert response.get_json() == {'detail': 'Invalid token'}

    def test_protected_route_handles_auth_service_unavailable(self, unauthenticated_client):
        with patch(
            'controllers.auth_controller.fetch_auth_user_data',
            side_effect=UpstreamUnavailable('down'),
        ):
            response = unauthenticated_client.get(
                '/api/projects',
                headers={'Authorization': 'Bearer any-token'},
            )

        assert response.status_code == 503
        assert response.get_json() == {'detail': 'Auth service unavailable'}


class TestSSORoutes:
    def test_sso_login_is_public(self, unauthenticated_client):
        response = unauthenticated_client.get('/api/sso/login')

        assert response.status_code == 200
        assert 'Authenticate via SSO' in response.get_data(as_text=True)

    def test_sso_exchange_returns_token_json(self, unauthenticated_client):
        with patch(
            'controllers.auth_controller._http_json_request',
            return_value=(200, '{"access_token":"abc"}', {'access_token': 'abc'}),
        ):
            response = unauthenticated_client.post(
                '/api/sso/exchange',
                json={'code': 'auth-code', 'redirect_uri': 'https://client.example.test/callback'},
            )

        assert response.status_code == 200
        assert response.get_json()['access_token'] == 'abc'

    def test_sso_refresh_returns_token_json(self, unauthenticated_client):
        with patch(
            'controllers.auth_controller._http_json_request',
            return_value=(200, '{"access_token":"fresh"}', {'access_token': 'fresh'}),
        ):
            response = unauthenticated_client.post(
                '/api/sso/refresh',
                json={'refresh_token': 'refresh-token'},
            )

        assert response.status_code == 200
        assert response.get_json()['access_token'] == 'fresh'

    def test_sso_callback_renders_access_token(self, unauthenticated_client):
        with patch(
            'controllers.auth_controller._http_json_request',
            return_value=(200, '{"access_token":"abc"}', {'access_token': 'abc'}),
        ):
            response = unauthenticated_client.get('/api/sso/callback?code=auth-code')

        assert response.status_code == 200
        assert 'Access Token' in response.get_data(as_text=True)
        assert 'abc' in response.get_data(as_text=True)
