"""User template API unit tests."""

from contextlib import contextmanager
from unittest.mock import patch

from conftest import assert_success_response
from models import UserTemplate, db


def _reset_db(app):
    with app.app_context():
        db.session.remove()
        db.drop_all()
        db.create_all()


@contextmanager
def _authenticated_client(app, email='test@example.com'):
    with patch('controllers.auth_controller.fetch_auth_user_data', return_value={
        'success': True,
        'data': {'email': email},
    }):
        with app.test_client() as test_client:
            test_client.environ_base['HTTP_AUTHORIZATION'] = 'Bearer test-token'
            yield test_client


class TestUserTemplateCreate:
    """User template creation tests."""

    def test_upload_user_template_persists_authenticated_user_email(self, app, sample_image_file):
        _reset_db(app)

        with _authenticated_client(app) as client:
            response = client.post(
                '/api/user-templates',
                data={
                    'name': 'Owned template',
                    'template_image': (sample_image_file, 'template.png'),
                },
                content_type='multipart/form-data',
            )

        data = assert_success_response(response)

        assert data['data']['user_email'] == 'test@example.com'
        assert data['data']['name'] == 'Owned template'


class TestUserTemplateList:
    """User template listing tests."""

    def test_list_user_templates_filters_by_authenticated_user_email(self, app):
        _reset_db(app)

        with app.app_context():
            db.session.add(UserTemplate(
                id='foreign-template',
                user_email='other@example.com',
                name='Foreign template',
                file_path='user-templates/foreign-template/template.png',
                file_size=123,
            ))
            db.session.add(UserTemplate(
                id='owned-template',
                user_email='test@example.com',
                name='Owned template',
                file_path='user-templates/owned-template/template.png',
                file_size=456,
            ))
            db.session.commit()

        with _authenticated_client(app) as client:
            response = client.get('/api/user-templates')

        data = assert_success_response(response)

        template_ids = [template['template_id'] for template in data['data']['templates']]
        user_emails = {template['user_email'] for template in data['data']['templates']}

        assert template_ids == ['owned-template']
        assert user_emails == {'test@example.com'}


class TestUserTemplateDelete:
    """User template deletion tests."""

    def test_delete_user_template_returns_404_for_other_users_template(self, app):
        _reset_db(app)

        with app.app_context():
            db.session.add(UserTemplate(
                id='foreign-template',
                user_email='other@example.com',
                name='Foreign template',
                file_path='user-templates/foreign-template/template.png',
                file_size=123,
            ))
            db.session.commit()

        with _authenticated_client(app) as client:
            response = client.delete('/api/user-templates/foreign-template')

        assert response.status_code == 404
