import joblib

lr = joblib.load("lr_model.pkl")
dt = joblib.load("dt_model.pkl")

g1 = int(input("Nhập điểm G1: "))
g2 = int(input("Nhập điểm G2: "))
absences = int(input("Nhập số buổi nghỉ: "))
studytime = int(input("Nhập thời gian học mỗi tuần (1-4): "))

X_input = [[g1, g2, absences, studytime]]

proba_lr = lr.predict_proba(X_input)[0][0]
proba_dt = dt.predict_proba(X_input)[0][0]

print(f"✅ Qua môn (xác suất rớt LR: {proba_lr*100:.2f}%)")
print(f"✅ Qua môn (xác suất rớt DT: {proba_dt*100:.2f}%)")