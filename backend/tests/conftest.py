"""Shared pytest fixtures for backend tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the backend package is importable before importing the app.
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

os.environ['TESTING'] = 'true'
os.environ['USE_MOCK_AI'] = 'true'
os.environ['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY', 'mock-api-key-for-testing')
os.environ['FLASK_ENV'] = 'testing'
os.environ['AUTH_SERVICE_URL'] = os.environ.get('AUTH_SERVICE_URL', 'http://auth.example.test/validate')
os.environ['SSO_AUTHORIZE_URL'] = os.environ.get('SSO_AUTHORIZE_URL', 'https://sso.example.test/authorize')
os.environ['SSO_TOKEN_URL'] = os.environ.get('SSO_TOKEN_URL', 'https://sso.example.test/token')
os.environ['SSO_CLIENT_ID'] = os.environ.get('SSO_CLIENT_ID', 'test-client-id')
os.environ['SSO_CLIENT_SECRET'] = os.environ.get('SSO_CLIENT_SECRET', 'test-client-secret')


@pytest.fixture(scope='session')
def app():
    """Create a Flask test application."""
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, 'test.db')
    os.environ['DATABASE_URL'] = f'sqlite:///{temp_db}'

    from app import create_app

    test_app = create_app()
    test_app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{temp_db}',
        'WTF_CSRF_ENABLED': False,
        'UPLOAD_FOLDER': temp_dir,
    })

    with test_app.app_context():
        from models import db
        db.create_all()

    yield test_app

    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def _reset_database(app):
    with app.app_context():
        from models import db
        db.session.rollback()
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()


@pytest.fixture(scope='function')
def client(app):
    """Authenticated client for most API tests."""
    with patch('controllers.auth_controller.fetch_auth_user_data', return_value={
        'success': True,
        'data': {
            'email': 'test@example.com',
            'ou_name': 'Test OU',
        },
    }):
        with app.test_client() as test_client:
            test_client.environ_base['HTTP_AUTHORIZATION'] = 'Bearer test-token'
            _reset_database(app)
            yield test_client
            with app.app_context():
                from models import db
                db.session.rollback()


@pytest.fixture(scope='function')
def unauthenticated_client(app):
    """Client without auth patching for middleware tests."""
    with app.test_client() as test_client:
        _reset_database(app)
        yield test_client
        with app.app_context():
            from models import db
            db.session.rollback()


@pytest.fixture(scope='function')
def db_session(app):
    """Database session fixture."""
    with app.app_context():
        from models import db
        db.create_all()
        yield db.session
        db.session.remove()
        db.drop_all()


@pytest.fixture
def sample_project(client):
    """Create a sample project via the API."""
    response = client.post('/api/projects', json={
        'creation_type': 'idea',
        'idea_prompt': 'Test PPT generation',
    })
    data = response.get_json()
    return data['data'] if data.get('success') else None


@pytest.fixture
def mock_ai_service():
    """Mock AI service to avoid external API calls."""
    with patch('services.ai_service.AIService') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance

        mock_instance.generate_outline.return_value = [
            {'title': 'Test page 1', 'points': ['Point 1', 'Point 2']},
            {'title': 'Test page 2', 'points': ['Point 3', 'Point 4']},
        ]
        mock_instance.flatten_outline.return_value = [
            {'title': 'Test page 1', 'points': ['Point 1', 'Point 2']},
            {'title': 'Test page 2', 'points': ['Point 3', 'Point 4']},
        ]
        mock_instance.generate_page_description.return_value = {
            'title': 'Test title',
            'text_content': ['Content 1', 'Content 2'],
            'layout_suggestion': 'Centered layout',
        }

        from PIL import Image

        test_image = Image.new('RGB', (1920, 1080), color='blue')
        mock_instance.generate_image.return_value = test_image

        yield mock_instance


@pytest.fixture
def temp_upload_dir():
    """Temporary upload directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image_file():
    """Create a simple sample PNG image."""
    import io
    from PIL import Image

    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


def assert_success_response(response, status_code=200):
    """Assert a successful JSON response."""
    assert response.status_code == status_code
    data = response.get_json()
    assert data is not None
    assert data.get('success') is True
    return data


def assert_error_response(response, expected_status=None):
    """Assert an error JSON response."""
    if expected_status:
        assert response.status_code == expected_status
    data = response.get_json()
    assert data is not None
    assert data.get('success') is False or 'error' in data
    return data
