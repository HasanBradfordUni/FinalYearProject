import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app.models import create_connection, create_tables, create_user

# Database path
db_path = os.path.join(os.path.dirname(__file__), 'app/static', 'placements.db')

# Ensure the directory exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Create connection and tables
print("Creating database connection...")
connection = create_connection(db_path)

def create_admin():
    if connection is None:
        print("\n✗ Error: Failed to create database connection!")
        print(f"Database path: {db_path}")
        print("Please check the path and permissions.")
        sys.exit(1)

    print("Creating database tables...")
    create_tables(connection)

    # Create admin user
    print("\nCreating admin user...")
    try:
        admin_id = create_user(
            connection=connection,
            username='HasanAk',
            email='akhtar.hasan@bradfordcft.org.uk',
            password='adminThis123!!',  # Change this after first login!
            role='admin'
        )

        print("\n" + "="*50)
        print("✓ Admin user created successfully!")
        print("="*50)
        print(f"User ID: {admin_id}")
        print("Username: AkhtarHa3")
        print("Email: hasan.akhtar@bradfordcft.org.uk")
        print("Password: Admin123")
        print("Role: admin")
        print("="*50)
        print("\n⚠️  IMPORTANT: Change the password after first login!")

    except Exception as e:
        print(f"\n✗ Error creating admin user: {e}")
        print("The admin user may already exist.")

if __name__ == '__main__':
    create_admin()
    connection.close()
    print("\nDatabase connection closed.")

