# Training the model
## Used datasets
https://www.kaggle.com/datasets/serhiibiruk/balloon-object-detection
## Used model
yolov8s.pt
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
4. Train model
```bash
yolo detect train data=..\..\balloon-dataset\Balloon\data.yaml model=yolo8s.pt
```