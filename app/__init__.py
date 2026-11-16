import os
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import Flask, jsonify, redirect, request, url_for
from flask_login import current_user
from werkzeug.middleware.proxy_fix import ProxyFix

from .extensions import csrf, db, login_manager, migrate


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY"),
        SQLALCHEMY_DATABASE_URI=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://localhost/tiffintrack"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 5,
            "pool_recycle": 1800,
        },
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_SECURE=True,
        REMEMBER_COOKIE_HTTPONLY=True,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        WTF_CSRF_TIME_LIMIT=3600,
        TIFFIN_PRICE=os.getenv("TIFFIN_PRICE", "80.00"),
        FULL_TIFFIN_PRICE=os.getenv("FULL_TIFFIN_PRICE", "80.00"),
        HALF_TIFFIN_PRICE=os.getenv("HALF_TIFFIN_PRICE", "60.00"),
        APP_TIMEZONE=os.getenv("APP_TIMEZONE", "Asia/Kolkata"),
        MAX_CONTENT_LENGTH=1024 * 1024,
        WA_BRIDGE_TOKEN=os.getenv("WA_BRIDGE_TOKEN", ""),
        SUMMARY_PUBLISHING_ENABLED=os.getenv("SUMMARY_PUBLISHING_ENABLED", "false").lower()
        == "true",
        # Compose binds the app to loopback and Nginx overwrites forwarded
        # headers, so exactly one trusted proxy can supply the client address.
        TRUST_PROXY=os.getenv("TRUST_PROXY", "false").lower() in {"1", "true", "yes"},
    )
    if test_config:
        app.config.update(test_config)
    if not app.config["SECRET_KEY"] or len(app.config["SECRET_KEY"]) < 32:
        raise RuntimeError("SECRET_KEY must be a random string of at least 32 characters")
    try:
        ZoneInfo(app.config["APP_TIMEZONE"])
        price = Decimal(str(app.config["TIFFIN_PRICE"]))
        if (
            not price.is_finite()
            or price <= 0
            or price.as_tuple().exponent < -2
            or price > 99999999
        ):
            raise ValueError
    except (ZoneInfoNotFoundError, ValueError, InvalidOperation) as exc:
        raise RuntimeError("Invalid APP_TIMEZONE or TIFFIN_PRICE") from exc
    from .domain.meal_items import portion_price

    for portion in ("standard", "full", "half"):
        portion_price(portion, app.config)
    if app.config["TRUST_PROXY"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=0, x_port=0, x_prefix=0)
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.session_protection = "strong"
    app.jinja_env.filters["money"] = lambda value: format(Decimal(str(value)), ".2f")
    csrf.init_app(app)
    from . import (
        models,  # noqa: F401
        whatsapp_models,  # noqa: F401
    )
    from .auth import auth_bp
    from .infrastructure import participants  # noqa: F401
    from .operations import operations_bp
    from .presentation.report_routes import report_bp
    from .routes import main_bp
    from .whatsapp_routes import whatsapp_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(whatsapp_bp)
    app.register_blueprint(operations_bp)
    app.register_blueprint(report_bp)

    @app.before_request
    def restrict_password_change():
        if (
            current_user.is_authenticated
            and current_user.must_change_password
            and request.endpoint not in {"auth.change_password", "auth.logout", "static", "health"}
        ):
            return redirect(url_for("auth.change_password"))

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' https://cdn.jsdelivr.net; script-src 'self' https://cdn.jsdelivr.net; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        if app.config["SESSION_COOKIE_SECURE"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/health")
    def health():
        try:
            db.session.execute(db.text("SELECT 1"))
            return jsonify(status="ok")
        except Exception:
            app.logger.exception("Database health check failed")
            return jsonify(status="unavailable"), 503

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()

    return app
