from flask import Flask
from flask_wtf import CSRFProtect
from flask_login import LoginManager
import os
from datetime import timedelta

csrf = CSRFProtect()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

    # Remember-me cookie configuration.
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=int(os.environ.get('REMEMBER_COOKIE_DAYS', '30')))
    app.config['REMEMBER_COOKIE_HTTPONLY'] = True
    app.config['REMEMBER_COOKIE_SECURE'] = bool(int(os.environ.get('REMEMBER_COOKIE_SECURE', '0')))

    # Optional SMTP configuration for temporary password emails.
    app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER')
    app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', '587'))
    app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
    app.config['MAIL_USE_TLS'] = bool(int(os.environ.get('MAIL_USE_TLS', '1')))

    # Initialize extensions
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'app.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # User loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from .models import get_user_by_id, create_connection
        db_path = os.path.join(os.path.dirname(__file__), 'static', 'placements.db')
        connection = create_connection(db_path)
        return get_user_by_id(connection, int(user_id))

    # Register blueprints
    from .routes import app as routes_blueprint
    app.register_blueprint(routes_blueprint)

    return app

