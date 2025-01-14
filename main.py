import os
import subprocess

def main():
    app_path = os.path.join("codes", "GUI", "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: The specified Streamlit app file '{app_path}' does not exist.")
        return

    try:
        print(f"Launching Streamlit app: {app_path}")
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the Streamlit app: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
