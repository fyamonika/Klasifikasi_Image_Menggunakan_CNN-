"""
Dashboard Streamlit - Klasifikasi Image OceanicLife Menggunakan CNN (MobileNetV2)

Jalankan:  streamlit run app.py

File yang dibutuhkan di folder output_oceaniclife/ (hasil notebook):
    oceaniclife_cnn_final.keras  (atau best_model.keras)
    class_names.json
    train.csv, validation.csv, test.csv
    history.json                 (dibuat dari sel tambahan, lihat penjelasan di chat)
Opsional: 2_class_examples.png, 3_image_properties.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ==========================================
# KONFIGURASI
# ==========================================
_BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = (
    os.path.join(_BASE, "output_oceaniclife")
    if os.path.isdir(os.path.join(_BASE, "output_oceaniclife"))
    else _BASE
)
IMG_SIZE = (160, 160)
DEFAULT_BASE_DIR = (
    r"C:\Users\User\OneDrive\Documents\Klasifikasi Image Menggunakan CNN\OceanicLife Dataset"
)
MODEL_CANDIDATES = [
    os.path.join(OUT_DIR, "oceaniclife_cnn_final.keras"),
    os.path.join(OUT_DIR, "best_model.keras"),
]
DEFAULT_CLASSES = [
    "Fish", "Coral", "Shark", "Octopuses",
    "Crustaceans", "Seaweed", "Sponges", "Other marine organisms",
]

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass  # sudah terinisialisasi pada rerun Streamlit; aman diabaikan
st.set_page_config(page_title="OceanicLife CNN Dashboard", page_icon="🌊", layout="wide")


# ==========================================
# LOADER
# ==========================================
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return tf.keras.models.load_model(p), p
    return None, None


@st.cache_data
def load_class_names():
    p = os.path.join(OUT_DIR, "class_names.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASSES


def _mtime(fname):
    """Waktu modifikasi file (0 jika tidak ada) -> dipakai agar cache ikut berubah saat file berubah."""
    p = os.path.join(OUT_DIR, fname)
    return os.path.getmtime(p) if os.path.exists(p) else 0


@st.cache_data
def _load_split(name, mtime):
    p = os.path.join(OUT_DIR, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_split(name):
    return _load_split(name, _mtime(f"{name}.csv"))


@st.cache_data
def _load_history(mtime):
    p = os.path.join(OUT_DIR, "history.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return None


def load_history():
    return _load_history(_mtime("history.json"))


def resolve_path(row, base_dir):
    """Cari file gambar: pakai 'filepath' dari CSV, jika tidak ada coba BASE_DIR + filename."""
    fp = str(row.get("filepath", ""))
    if fp and os.path.exists(fp):
        return fp
    alt = os.path.join(base_dir, str(row.get("filename", "")))
    return alt if os.path.exists(alt) else None


# ==========================================
# PREDIKSI & EVALUASI
# ==========================================
def preprocess(img):
    # Normalisasi [-1,1] sudah dilakukan layer Rescaling di dalam model -> input 0-255.
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32)


def predict_one(model, img):
    return model.predict(np.expand_dims(preprocess(img), 0), verbose=0)[0]


@st.cache_data(show_spinner="Mengevaluasi model pada data uji...")
def evaluate_test(base_dir, n_classes, eval_mtime=0):
    # Prioritas 1: hasil evaluasi yang sudah dihitung di notebook (tanpa perlu gambar)
    ep = os.path.join(OUT_DIR, "eval_results.json")
    if os.path.exists(ep):
        with open(ep, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "y_true": np.array(d["y_true"]),
            "y_pred": np.array(d["y_pred"]),
            "missing": 0,
            "n": len(d["y_true"]),
        }
    # Prioritas 2: hitung ulang dari gambar data uji
    model, _ = load_model()
    te = load_split("test")
    if model is None or te is None:
        return None
    arrays, y_true, missing = [], [], 0
    for _, row in te.iterrows():
        p = resolve_path(row, base_dir)
        if p is None:
            missing += 1
            continue
        try:
            arrays.append(preprocess(Image.open(p)))
            y_true.append(int(row["label_id"]))
        except Exception:
            missing += 1
    if not arrays:
        return {"missing": missing, "n": 0}
    probs = model.predict(np.stack(arrays), batch_size=32, verbose=0)
    return {
        "y_true": np.array(y_true),
        "y_pred": probs.argmax(axis=1),
        "missing": missing,
        "n": len(y_true),
    }


# ==========================================
# ANALISIS OTOMATIS
# ==========================================
def show_analysis(text):
    st.info("**Analisis:** " + text)


def analisis_dataset(n_img, n_cls, counts, split_sizes):
    big, small = counts.idxmax(), counts.idxmin()
    ratio = counts.max() / max(counts.min(), 1)
    tr, va, te = split_sizes
    tot = max(tr + va + te, 1)
    txt = (
        f"Dataset berisi {n_img} gambar berlabel dalam {n_cls} kelas. Pembagian data "
        f"{tr / tot * 100:.0f}% latih, {va / tot * 100:.0f}% validasi, dan {te / tot * 100:.0f}% uji "
        f"(stratified, sehingga proporsi kelas di tiap subset kurang lebih sama). "
        f"Kelas terbanyak adalah {big} ({counts.max()} gambar) dan tersedikit adalah {small} "
        f"({counts.min()} gambar), rasio {ratio:.1f}:1. "
    )
    if ratio >= 3:
        txt += ("Ketimpangan ini cukup besar, sehingga model berisiko lebih bias ke kelas mayoritas; "
                "karena itu notebook memakai class weight 'balanced' saat pelatihan.")
    else:
        txt += "Distribusi kelas relatif seimbang sehingga risiko bias ke satu kelas kecil."
    return txt


def analisis_distribusi(counts):
    top3 = ", ".join(f"{k} ({v})" for k, v in counts.sort_values(ascending=False).head(3).items())
    kosong = list(counts[counts == 0].index)
    txt = f"Tiga kelas dengan data terbanyak: {top3}. "
    if kosong:
        txt += f"Kelas tanpa data: {', '.join(kosong)} - model tidak dapat mempelajarinya. "
    sedikit = list(counts[(counts > 0) & (counts < 10)].index)
    if sedikit:
        txt += (f"Kelas dengan kurang dari 10 gambar ({', '.join(sedikit)}) kemungkinan sulit dikenali "
                "dan metrik per kelasnya kurang stabil.")
    return txt


def analisis_training(h):
    ep = len(h)
    best = int(h["val_loss"].idxmin())
    tr_acc, va_acc = h["accuracy"].iloc[-1], h["val_accuracy"].iloc[-1]
    gap = tr_acc - va_acc
    txt = (
        f"Pelatihan berjalan {ep} epoch. Val loss terendah ({h['val_loss'].min():.4f}) tercapai pada "
        f"epoch {best + 1}. Di epoch terakhir akurasi latih {tr_acc * 100:.2f}% dan validasi "
        f"{va_acc * 100:.2f}% (selisih {gap * 100:.2f} poin). "
    )
    if gap > 0.10:
        txt += ("Selisih yang besar menandakan overfitting: model hafal data latih tetapi kurang "
                "menggeneralisasi. Tambah data/augmentasi atau perbesar dropout.")
    elif va_acc < 0.6 and tr_acc < 0.6:
        txt += "Akurasi kedua kurva rendah (underfitting); pertimbangkan epoch lebih banyak atau fine-tuning."
    else:
        txt += "Kurva latih dan validasi berdekatan sehingga model belajar dengan cukup baik dan stabil."
    return txt


def analisis_cm(cm, names):
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    supported = cm.sum(axis=1) > 0
    r = pd.Series(recall, index=names)[supported]
    best, worst = r.idxmax(), r.idxmin()
    off = cm.copy().astype(int)
    np.fill_diagonal(off, 0)
    txt = (f"Diagonal utama adalah prediksi benar. Kelas paling akurat: {best} (recall {r.max() * 100:.1f}%), "
           f"paling sulit: {worst} (recall {r.min() * 100:.1f}%). ")
    if off.max() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        txt += (f"Kesalahan terbanyak: {names[i]} salah diprediksi sebagai {names[j]} ({off.max()} gambar), "
                "biasanya karena bentuk atau latar yang mirip.")
    else:
        txt += "Tidak ada kesalahan klasifikasi pada data uji."
    return txt


def analisis_metrik(acc, mp, mr, mf, wf):
    txt = (f"Akurasi {acc * 100:.2f}%, sedangkan macro F1 {mf * 100:.2f}% dan weighted F1 {wf * 100:.2f}%. ")
    if acc - mf > 0.05:
        txt += ("Akurasi lebih tinggi dari macro F1, artinya performa kelas mayoritas lebih baik daripada "
                "kelas minoritas. Perhatikan tabel per kelas untuk kelas yang lemah. ")
    else:
        txt += "Macro F1 dekat dengan akurasi sehingga performa antar kelas cukup merata. "
    if mp > mr + 0.03:
        txt += "Precision rata-rata lebih tinggi dari recall: model hati-hati tetapi melewatkan sebagian objek."
    elif mr > mp + 0.03:
        txt += "Recall rata-rata lebih tinggi dari precision: model cenderung terlalu sering memberi label kelas tertentu."
    else:
        txt += "Precision dan recall seimbang."
    return txt


def analisis_prediksi(probs, names):
    order = np.argsort(probs)[::-1]
    t1, t2 = order[0], order[1]
    c1, c2 = probs[t1] * 100, probs[t2] * 100
    txt = f"Model memprediksi **{names[t1]}** dengan confidence {c1:.2f}%. "
    if c1 >= 80:
        txt += "Confidence tinggi, prediksi dapat diandalkan. "
    elif c1 >= 50:
        txt += "Confidence sedang; pertimbangkan memeriksa kandidat kedua. "
    else:
        txt += "Confidence rendah, gambar mungkin ambigu atau di luar kelas yang dipelajari. "
    txt += f"Kandidat kedua: {names[t2]} ({c2:.2f}%), selisih {c1 - c2:.2f} poin."
    return txt


# ==========================================
# MEMUAT DATA
# ==========================================
model, model_path = load_model()
class_names = load_class_names()
n_cls = len(class_names)
tr_df, va_df, te_df = load_split("train"), load_split("validation"), load_split("test")
hist = load_history()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    base_dir = st.text_input("Folder dataset gambar", DEFAULT_BASE_DIR,
                             help="Dipakai jika path di CSV tidak ditemukan (mis. dijalankan di komputer lain).")
    st.caption(f"Model: `{model_path or 'tidak ditemukan'}`")
    st.caption(f"Input: {IMG_SIZE[0]}x{IMG_SIZE[1]} RGB • {n_cls} kelas")

# ==========================================
# HEADER
# ==========================================
st.title("🌊 Dashboard Klasifikasi Citra OceanicLife")
st.markdown(
    """
Proyek ini mengklasifikasikan gambar biota laut ke dalam **8 kelas** (Fish, Coral, Shark, Octopuses,
Crustaceans, Seaweed, Sponges, dan Other marine organisms) menggunakan **Convolutional Neural Network**
berbasis transfer learning **MobileNetV2** (bobot ImageNet) dengan augmentasi data, class weight,
dan callback *EarlyStopping*, *ModelCheckpoint*, serta *ReduceLROnPlateau*.
Dashboard ini menampilkan ringkasan dataset, hasil pelatihan, evaluasi, dan uji prediksi gambar baru.
"""
)

if model is None:
    st.error("Model tidak ditemukan di folder `output_oceaniclife/`. Jalankan notebook sampai tahap penyimpanan model.")
    st.stop()

# ==========================================
# 1. RINGKASAN DATASET
# ==========================================
st.header("1. Ringkasan Dataset")
if tr_df is None or va_df is None or te_df is None:
    st.warning("File train.csv / validation.csv / test.csv belum ada di `output_oceaniclife/`.")
    all_df = None
else:
    all_df = pd.concat([tr_df, va_df, te_df], ignore_index=True)
    counts = all_df["label"].value_counts().reindex(class_names, fill_value=0)
    n_img, sizes = len(all_df), (len(tr_df), len(va_df), len(te_df))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Jumlah kelas", n_cls)
    c2.metric("Jumlah gambar", n_img)
    c3.metric("Data latih", f"{sizes[0]} ({sizes[0] / n_img * 100:.0f}%)")
    c4.metric("Data validasi", f"{sizes[1]} ({sizes[1] / n_img * 100:.0f}%)")
    c5.metric("Data uji", f"{sizes[2]} ({sizes[2] / n_img * 100:.0f}%)")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.pie(sizes, labels=["Latih", "Validasi", "Uji"], autopct="%1.1f%%",
           colors=["#2a9d8f", "#e9c46a", "#e76f51"], startangle=90)
    ax.set_title("Proporsi Pembagian Data")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.pyplot(fig)
    with col_b:
        st.write("")
        show_analysis(analisis_dataset(n_img, n_cls, counts, sizes))
    plt.close(fig)

    # ---------- 2. DISTRIBUSI ----------
    st.header("2. Distribusi Data per Kelas")
    dist = pd.DataFrame({
        "Latih": tr_df["label"].value_counts().reindex(class_names, fill_value=0),
        "Validasi": va_df["label"].value_counts().reindex(class_names, fill_value=0),
        "Uji": te_df["label"].value_counts().reindex(class_names, fill_value=0),
    })
    fig, ax = plt.subplots(figsize=(10, 4))
    dist.plot(kind="bar", stacked=True, ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_ylabel("Jumlah gambar")
    ax.set_xlabel("")
    ax.set_title("Jumlah Gambar per Kelas (Latih / Validasi / Uji)")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(dist.sum(axis=1)):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_analysis(analisis_distribusi(counts))

    for fname, title, cap in [
        ("2_class_examples.png", "Contoh Gambar Tiap Kelas",
         "Contoh visual memperlihatkan variasi sudut pandang, pencahayaan bernuansa biru/hijau, dan latar belakang. "
         "Kelas berbentuk mirip (Fish-Shark, Coral-Sponges) berpotensi tertukar."),
        ("3_image_properties.png", "Properti Gambar Dataset",
         "Ukuran gambar tidak seragam dan mode warna bisa bercampur, sehingga semua gambar di-resize ke "
         f"{IMG_SIZE[0]}x{IMG_SIZE[1]} dan dibaca sebagai RGB."),
    ]:
        pth = os.path.join(OUT_DIR, fname)
        if os.path.exists(pth):
            with st.expander(title):
                st.image(pth, width="stretch")
                show_analysis(cap)

# ==========================================
# 3. HASIL PELATIHAN
# ==========================================
st.header("3. Hasil Pelatihan")
if hist is None:
    st.warning("`history.json` belum ada. Jalankan sel penyimpanan history di notebook (lihat petunjuk).")
else:
    ep = np.arange(1, len(hist) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, hist["accuracy"], marker="o", label="Latih")
    axes[0].plot(ep, hist["val_accuracy"], marker="o", label="Validasi")
    axes[0].set_title("Akurasi per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Akurasi")
    axes[1].plot(ep, hist["loss"], marker="o", label="Latih")
    axes[1].plot(ep, hist["val_loss"], marker="o", label="Validasi")
    axes[1].set_title("Loss per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    for a in axes:
        a.grid(alpha=0.3); a.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_analysis(analisis_training(hist))

# ==========================================
# 4-5. EVALUASI
# ==========================================
st.header("4. Evaluasi Model pada Data Uji")
ev = evaluate_test(base_dir, n_cls, _mtime("eval_results.json")) if te_df is not None else None
if ev is None or ev.get("n", 0) == 0:
    st.warning("Evaluasi belum tersedia: `eval_results.json` tidak ditemukan dan gambar data uji "
               "tidak bisa dibaca. Jalankan sel penyimpanan evaluasi di notebook, lalu upload "
               "`eval_results.json` ke repo.")
else:
    if ev["missing"]:
        st.caption(f"⚠️ {ev['missing']} gambar uji tidak ditemukan/rusak dan dilewati.")
    y_true, y_pred = ev["y_true"], ev["y_pred"]
    labels = list(range(n_cls))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    mode = st.radio("Tampilan", ["Jumlah", "Dinormalisasi (per baris)"], horizontal=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if mode == "Jumlah":
        data, fmt = cm, "d"
    else:
        data, fmt = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), ".2f"
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix - Data Uji")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_analysis(analisis_cm(cm, class_names))

    # Metrik
    st.subheader("Accuracy, Precision, Recall, F1-score")
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Precision (macro)", f"{mp * 100:.2f}%")
    m3.metric("Recall (macro)", f"{mr * 100:.2f}%")
    m4.metric("F1-score (macro)", f"{mf * 100:.2f}%")

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rep = pd.DataFrame({"Kelas": class_names, "Precision": p, "Recall": r, "F1-score": f, "Support": s})
    st.dataframe(rep.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                 width="stretch", hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(n_cls); w = 0.27
    ax.bar(x - w, p, w, label="Precision"); ax.bar(x, r, w, label="Recall"); ax.bar(x + w, f, w, label="F1-score")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_title("Metrik per Kelas"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_analysis(analisis_metrik(acc, mp, mr, mf, wf))

# ==========================================
# 6. PREDIKSI GAMBAR BARU (REAL-TIME)
# ==========================================
st.header("5. Uji Prediksi Gambar Baru")
src = st.radio("Sumber gambar", ["Upload", "Kamera"], horizontal=True)
file = (st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if src == "Upload" else st.camera_input("Ambil foto"))

if file is not None:
    try:
        image = Image.open(file)
    except Exception as e:
        st.error(f"Gambar tidak dapat dibuka: {e}")
        st.stop()

    with st.spinner("Memprediksi..."):
        probs = predict_one(model, image)
    top = int(np.argmax(probs))

    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Gambar input", width="stretch")
    with c2:
        st.subheader("Hasil Prediksi")
        st.metric("Kelas terdeteksi", class_names[top])
        st.metric("Confidence", f"{probs[top] * 100:.2f}%")

    pdf = (pd.DataFrame({"Kelas": class_names, "Probabilitas (%)": probs * 100})
           .sort_values("Probabilitas (%)", ascending=False))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#2a9d8f" if k == class_names[top] else "#b0bec5" for k in pdf["Kelas"]]
    ax.barh(pdf["Kelas"][::-1], pdf["Probabilitas (%)"][::-1], color=colors[::-1])
    ax.set_xlabel("Probabilitas (%)"); ax.set_xlim(0, 100); ax.set_title("Probabilitas Setiap Kelas")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    show_analysis(analisis_prediksi(probs, class_names))
else:
    st.info("Unggah gambar atau ambil foto untuk melihat prediksi secara langsung.")
