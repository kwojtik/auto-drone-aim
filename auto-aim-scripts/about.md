# Scripts for aim assist
## Activating
1. If the env directory is not created
    ```bash
    python -m venv .venv
    ```
2. Activate virtual environment
    ```bash
    .venv\Scripts\activate
    ```
3. Download requirements
    ```bash
    pip install -r requirements.txt
    ```
        - if additionaL requirements are added, use command
    ```bash
    pip freeze > requirements.txt
    ```
4. Run script (example)
    ```bash
    python main.py --model ..\..\models\TestModel\my_model.pt --source webcam0 --resolution 1280x720
    ```