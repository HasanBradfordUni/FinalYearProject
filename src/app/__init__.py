from flask import Flask
from flask_wtf import CSRFProtect
from flask_login import LoginManager
import os

csrf = CSRFProtect()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

