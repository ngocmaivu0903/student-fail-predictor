# Dự đoán nguy cơ rớt môn

## Hướng dẫn chạy trên Google Colab

### Bước 1: Tải dataset
Chạy ô sau trong Colab để tải dữ liệu:
```python
!wget -O student-mat.csv https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv
```

### Bước 2: Cài thư viện cần thiết
```python
!pip install pandas scikit-learn joblib
```

### Bước 3: Huấn luyện mô hình
```python
!python3 model_training.py
```

### Bước 4: Chạy demo CLI
```python
!python3 predict_cli.py
```