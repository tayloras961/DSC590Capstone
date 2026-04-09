from werkzeug.security import generate_password_hash

DEFAULT_USERS = {
    "admin": {"password_hash": generate_password_hash("admin123"), "role": "admin"},
    "user": {"password_hash": generate_password_hash("user123"), "role": "user"},
}
