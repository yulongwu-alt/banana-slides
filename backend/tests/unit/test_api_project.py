"""Project API unit tests."""

import pytest

from conftest import assert_success_response
from models import Project


class TestProjectCreate:
    """Project creation tests."""

    def test_create_project_idea_mode(self, client):
        response = client.post('/api/projects', json={
            'creation_type': 'idea',
            'idea_prompt': 'Generate an AI presentation',
        })

        data = assert_success_response(response, 201)
        assert 'project_id' in data['data']
        assert data['data']['user_email'] == 'test@example.com'
        assert data['data']['status'] == 'DRAFT'

        detail = assert_success_response(client.get(f"/api/projects/{data['data']['project_id']}"))
        assert detail['data']['user_email'] == 'test@example.com'

    def test_create_project_outline_mode(self, client):
        response = client.post('/api/projects', json={
            'creation_type': 'outline',
            'outline': [
                {'title': 'Page 1', 'points': ['Point 1']},
                {'title': 'Page 2', 'points': ['Point 2']},
            ],
        })

        data = assert_success_response(response, 201)
        assert 'project_id' in data['data']

    def test_create_project_missing_type(self, client):
        response = client.post('/api/projects', json={
            'idea_prompt': 'Test',
        })

        assert response.status_code in [400, 422]

    def test_create_project_invalid_type(self, client):
        response = client.post('/api/projects', json={
            'creation_type': 'invalid_type',
            'idea_prompt': 'Test',
        })

        assert response.status_code in [400, 422]


class TestProjectGet:
    """Project retrieval tests."""

    def test_get_project_success(self, client, sample_project):
        if not sample_project:
            pytest.skip("Project creation failed")

        project_id = sample_project['project_id']
        response = client.get(f'/api/projects/{project_id}')

        data = assert_success_response(response)
        assert data['data']['project_id'] == project_id

    def test_get_project_filters_by_authenticated_user_email(self, client, db_session):
        foreign_project = Project(
            creation_type='idea',
            user_email='other@example.com',
            idea_prompt='Foreign project',
            status='DRAFT',
        )
        db_session.add(foreign_project)
        db_session.commit()

        response = client.get(f'/api/projects/{foreign_project.id}')

        assert response.status_code == 404

    def test_get_project_not_found(self, client):
        response = client.get('/api/projects/non-existent-id')
        assert response.status_code == 404

    def test_get_project_invalid_id_format(self, client):
        response = client.get('/api/projects/invalid!@#$%id')
        assert response.status_code in [400, 404]


class TestProjectUpdate:
    """Project update tests."""

    def test_update_project_status(self, client, sample_project):
        if not sample_project:
            pytest.skip("Project creation failed")

        project_id = sample_project['project_id']
        response = client.put(f'/api/projects/{project_id}', json={
            'status': 'GENERATING',
        })

        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['data']['user_email'] == 'test@example.com'


class TestProjectList:
    """Project listing tests."""

    def test_list_projects_filters_by_authenticated_user_email(self, client, db_session):
        db_session.add(Project(
            creation_type='idea',
            user_email='other@example.com',
            idea_prompt='Foreign project',
            status='DRAFT',
        ))
        db_session.commit()

        own_response = client.post('/api/projects', json={
            'creation_type': 'idea',
            'idea_prompt': 'Owned project',
        })
        own_project_id = assert_success_response(own_response, 201)['data']['project_id']

        response = client.get('/api/projects')

        data = assert_success_response(response)
        project_ids = [project['project_id'] for project in data['data']['projects']]
        user_emails = {project['user_email'] for project in data['data']['projects']}

        assert own_project_id in project_ids
        assert user_emails == {'test@example.com'}


class TestProjectDelete:
    """Project deletion tests."""

    def test_delete_project_success(self, client, sample_project):
        if not sample_project:
            pytest.skip("Project creation failed")

        project_id = sample_project['project_id']
        response = client.delete(f'/api/projects/{project_id}')

        assert_success_response(response)

        get_response = client.get(f'/api/projects/{project_id}')
        assert get_response.status_code == 404

    def test_delete_project_not_found(self, client):
        response = client.delete('/api/projects/non-existent-id')
        assert response.status_code == 404
