from pathlib import Path
import joblib
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(page_title="TP1 | Prediction Models", page_icon="🌿", layout="centered")

@st.cache_resource
def load_models():
    required = [
        "diabetes_slr.joblib", "diabetes_mlr.joblib", "diabetes_info.joblib",
        "iris_slr.joblib", "iris_mlr.joblib", "iris_classifier.joblib", "iris_info.joblib"
    ]
    missing = [f for f in required if not (MODELS / f).exists()]
    if missing:
        raise FileNotFoundError("Model files missing: " + ", ".join(missing) + ". Run: python train_models.py")
    return {Path(f).stem: joblib.load(MODELS / f) for f in required}

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown("<h1 style='text-align:center'>🌿 TP1 – Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#64748b'>واجهة لتشغيل المودالات على قيم جديدة وعرض التنبؤ فقط</p>", unsafe_allow_html=True)
st.divider()

choice = st.selectbox(
    "اختاري المودال الذي تريدين تشغيله",
    ["تصنيف نوع الزهرة (Iris Classification)", "تنبؤ تطور السكري – Simple Linear Regression", "تنبؤ تطور السكري – Multiple Linear Regression", "تنبؤ طول الكأس – Iris Simple Regression", "تنبؤ طول الكأس – Iris Multiple Regression"]
)

# Flower species classification: new measurements -> predicted class + corresponding flower photo.
if choice == "تصنيف نوع الزهرة (Iris Classification)":
    st.subheader("أدخلي قياسات الزهرة الجديدة")
    info = models["iris_info"]
    names = ["Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
    cols = st.columns(2)
    vals = []
    for i, name in enumerate(names):
        with cols[i % 2]:
            vals.append(st.number_input(name, min_value=0.0, max_value=10.0, value=float(info["mean"][i]), step=0.1, format="%.2f"))
    if st.button("تشغيل المودال والتنبؤ", type="primary", use_container_width=True):
        sample = np.array(vals, dtype=float).reshape(1, -1)
        clf = models["iris_classifier"]
        pred = int(clf.predict(sample)[0])
        probabilities = clf.predict_proba(sample)[0]
        species = info["target_names"][pred]
        st.success(f"النوع المتوقع: {species.title()}")
        st.markdown("#### صورة توضيحية للنوع المتنبأ به")
        flower_images = {
            "setosa": "https://commons.wikimedia.org/wiki/Special:FilePath/Wild_iris_flower_iris_setosa.jpg",
            "versicolor": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_versicolor_FWS.jpg",
            "virginica": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_virginica.jpg",
        }
        st.image(flower_images[species], caption=f"Iris {species.title()} – صورة مرجعية للنوع المتوقع", use_container_width=True)
        st.caption("الصورة مرجعية للتعرّف على النوع وليست صورة للزهرة التي تم قياسها. يلزم الاتصال بالإنترنت لعرض الصور.")
        st.markdown("**احتمالات المودال:**")
        for label, prob in zip(info["target_names"], probabilities):
            st.write(f"{label.title()}: {prob * 100:.1f}%")

elif choice == "تنبؤ تطور السكري – Simple Linear Regression":
    st.subheader("أدخلي قيمة BMI")
    info = models["diabetes_info"]
    st.info("تنبيه: بيانات Diabetes في Scikit-learn مُحوّلة ومُعيّرة (normalized). أدخلي قيمة ضمن المجال المعروض؛ هذه ليست قيمة BMI المعتادة مثل 25 أو 30.")
    bmi = st.number_input("BMI (normalized)", min_value=float(info["min"][2]), max_value=float(info["max"][2]), value=float(info["mean"][2]), step=0.01, format="%.4f")
    if st.button("تشغيل المودال والتنبؤ", type="primary", use_container_width=True):
        result = float(models["diabetes_slr"].predict(np.array([[bmi]]))[0])
        st.success(f"القيمة المتنبأ بها لتطور المرض: {result:.2f}")
        st.caption("هذه قيمة تنبؤية إحصائية من مجموعة Diabetes التجريبية، وليست تشخيصاً طبياً.")

elif choice == "تنبؤ تطور السكري – Multiple Linear Regression":
    st.subheader("أدخلي الخصائص العشر")
    info = models["diabetes_info"]
    st.info("تنبيه: خصائص مجموعة Diabetes مُحوّلة ومُعيّرة في Scikit-learn. القيم الافتراضية هي متوسطات البيانات، ويمكنك تغييرها.")
    cols = st.columns(2)
    values = []
    for i, name in enumerate(info["feature_names"]):
        with cols[i % 2]:
            values.append(st.number_input(name, min_value=float(info["min"][i]), max_value=float(info["max"][i]), value=float(info["mean"][i]), step=0.01, format="%.4f", key=f"db_{name}"))
    if st.button("تشغيل المودال والتنبؤ", type="primary", use_container_width=True):
        result = float(models["diabetes_mlr"].predict(np.array(values).reshape(1, -1))[0])
        st.success(f"القيمة المتنبأ بها لتطور المرض: {result:.2f}")
        st.caption("هذه قيمة تنبؤية إحصائية وليست تشخيصاً طبياً.")

elif choice == "تنبؤ طول الكأس – Iris Simple Regression":
    st.subheader("أدخلي طول البتلة")
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=3.75, step=0.1)
    if st.button("تشغيل المودال والتنبؤ", type="primary", use_container_width=True):
        result = float(models["iris_slr"].predict(np.array([[petal_length]]))[0])
        st.success(f"Sepal length المتوقع: {result:.2f} cm")
        st.caption("هذا المودال يتنبأ بقياس عددي فقط، ولا يحدد نوع الزهرة.")

else:
    st.subheader("أدخلي ثلاث قياسات")
    labels = ["Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"]
    defaults = [3.05, 3.76, 1.20]
    cols = st.columns(3)
    values = []
    for i, label in enumerate(labels):
        with cols[i]:
            values.append(st.number_input(label, min_value=0.0, max_value=10.0, value=defaults[i], step=0.1))
    if st.button("تشغيل المودال والتنبؤ", type="primary", use_container_width=True):
        result = float(models["iris_mlr"].predict(np.array(values).reshape(1, -1))[0])
        st.success(f"Sepal length المتوقع: {result:.2f} cm")
        st.caption("هذا المودال يتنبأ بقياس عددي فقط، ولا يحدد نوع الزهرة.")

st.divider()
st.caption("TP1 · Linear Regression & Iris Classification · Models trained from Scikit-learn datasets")
