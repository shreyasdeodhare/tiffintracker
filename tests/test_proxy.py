from flask import request

from app import create_app


def test_trusted_proxy_uses_forwarded_client_ip():
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "t" * 48,
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "SQLALCHEMY_ENGINE_OPTIONS": {},
            "TRUST_PROXY": True,
        }
    )
    app.add_url_rule("/test-client-ip", view_func=lambda: request.remote_addr)

    response = app.test_client().get("/test-client-ip", headers={"X-Forwarded-For": "203.0.113.10"})

    assert response.get_data(as_text=True) == "203.0.113.10"
