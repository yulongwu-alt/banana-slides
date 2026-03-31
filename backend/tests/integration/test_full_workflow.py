"""End-to-end API workflow tests."""

from conftest import assert_success_response


class TestFullWorkflow:
    """Workflow coverage for core project endpoints."""

    def test_create_project_and_get_details(self, client):
        create_response = client.post('/api/projects', json={
            'creation_type': 'idea',
            'idea_prompt': 'Generate a three-page quantum computing presentation',
        })

        data = assert_success_response(create_response, 201)
        project_id = data['data']['project_id']
        assert data['data']['user_email'] == 'test@example.com'

        get_response = client.get(f'/api/projects/{project_id}')
        data = assert_success_response(get_response)
        assert data['data']['project_id'] == project_id
        assert data['data']['user_email'] == 'test@example.com'
        assert data['data']['status'] == 'DRAFT'

    def test_template_upload_workflow(self, client, sample_image_file):
        create_response = client.post('/api/projects', json={
            'creation_type': 'idea',
            'idea_prompt': 'Test template upload',
        })

        data = assert_success_response(create_response, 201)
        project_id = data['data']['project_id']

        upload_response = client.post(
            f'/api/projects/{project_id}/template',
            data={'template_image': (sample_image_file, 'template.png')},
            content_type='multipart/form-data',
        )

        assert upload_response.status_code in [200, 201]

    def test_project_lifecycle(self, client):
        create_response = client.post('/api/projects', json={
            'creation_type': 'idea',
            'idea_prompt': 'Lifecycle test',
        })
        data = assert_success_response(create_response, 201)
        project_id = data['data']['project_id']

        get_response = client.get(f'/api/projects/{project_id}')
        assert_success_response(get_response)

        delete_response = client.delete(f'/api/projects/{project_id}')
        assert_success_response(delete_response)

        verify_response = client.get(f'/api/projects/{project_id}')
        assert verify_response.status_code == 404


class TestAPIErrorHandling:
    """Basic API error handling checks."""

    def test_invalid_json_body(self, client):
        response = client.post(
            '/api/projects',
            data='invalid json',
            content_type='application/json',
        )

        assert response.status_code in [400, 415, 422]

    def test_missing_required_fields(self, client):
        response = client.post('/api/projects', json={})
        assert response.status_code in [400, 422]

    def test_method_not_allowed(self, client):
        response = client.patch('/api/projects')
        assert response.status_code in [404, 405]


class TestConcurrentRequests:
    """Basic concurrent-like request coverage."""

    def test_multiple_project_creation(self, client):
        project_ids = []

        for i in range(3):
            response = client.post('/api/projects', json={
                'creation_type': 'idea',
                'idea_prompt': f'Concurrent test project {i}',
            })

            data = assert_success_response(response, 201)
            project_ids.append(data['data']['project_id'])

        assert len(set(project_ids)) == 3

        for project_id in project_ids:
            client.delete(f'/api/projects/{project_id}')
