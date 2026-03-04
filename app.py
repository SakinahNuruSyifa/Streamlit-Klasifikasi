import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from streamlit_option_menu import option_menu

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Klasifikasi Kategori Berita",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background-color: #f7f8fc;
}

/* SIDEBAR CONTENT */
section[data-testid="stSidebar"] > div {
    padding-top: 10px;
}

/* OPTION MENU FIX */
.nav-link {
    color: #3f51b5 !important;
}
.nav-link-selected {
    background-color: #e8ebff !important;
    color: #3f51b5 !important;
}

/* MAIN PAGE WIDTH */
.block-container {
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* BUTTON STYLE */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.card h4 {
    margin-bottom: 10px;
    color: #57564F;
}
.card p, .card li {
    font-size: 14px;
    color: #444;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =====================
# LOAD MODEL & TOKENIZER
# =====================
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")
    model.eval()
    return model, tokenizer

@st.cache_data
def load_label_map():
    with open("label_map.json", "r") as f:
        return json.load(f)

model, tokenizer = load_model()
label_map = load_label_map()

# =====================
# SIDEBAR MENU
# =====================
with st.sidebar:
    st.markdown("""
    <div style="padding:10px;">
        <h3>📰 NEWS DASHBOARD</h3>
        <p style="color:#666; font-size:13px;">
            Sistem Informasi Kategori Berita
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Klasifikasi Berita"],
        icons=["house", "newspaper"],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#f7f8fc",
            },
            "icon": {
                "color": "#3f51b5",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "15px",
                "font-weight": "600",
                "margin": "4px",
                "border-radius": "8px",
            },
            "nav-link-selected": {
                "background-color": "#e8ebff",
                "color": "#3f51b5",
            },
        }
    )

# =====================
# HALAMAN BERANDA
# =====================
if selected == "Beranda":
    st.markdown(
        "<h1 style='text-align:center;'>Sistem Klasifikasi Kategori Berita</h1>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("""
    <div class="card">
        <h4>📰 Tentang Sistem</h4>
        <p>
        Sistem Klasifikasi Kategori Berita merupakan web
        yang dirancang untuk mengelompokkan teks berita berbahasa Indonesia
        ke dalam kategori tertentu secara otomatis.
        </p>
        <p>
        Sistem ini dikembangkan menggunakan <i>Natural Language Processing (NLP)</i>
        dengan memanfaatkan model <b>IndoBERT</b>, sehingga mampu memahami konteks
        dan makna teks berita secara lebih akurat dibandingkan metode konvensional.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🎯 Tujuan Pengembangan Sistem</h4>
        <ul>
            <li>Mengimplementasikan model IndoBERT untuk klasifikasi kategori berita</li>
            <li>Menyediakan dashboard interaktif sebagai media implementasi penelitian</li>
            <li>Menampilkan hasil prediksi kategori beserta tingkat kepercayaannya</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>⚙️ Fitur Utama Sistem</h4>
        <ul>
            <li>Klasifikasi otomatis teks berita berbahasa Indonesia</li>
            <li>Menampilkan kategori berita hasil prediksi</li>
            <li>Menyediakan confidence score untuk setiap prediksi</li>
            <li>Visualisasi probabilitas setiap kategori</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🧠 Metode dan Teknologi</h4>
        <ul>
            <li>Model Transformer IndoBERT</li>
            <li>Bahasa pemrograman Python</li>
            <li>Framework Streamlit</li>
            <li>Library Deep Learning PyTorch</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🎓 Manfaat Sistem</h4>
        <ul>
            <li>Membantu proses pengelompokan berita secara otomatis</li>
            <li>Menjadi media implementasi dan evaluasi model NLP</li>
            <li>Mendukung penelitian akademik di bidang klasifikasi teks</li>
            <li>Menjadi referensi pengembangan sistem serupa di masa depan</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📖 Panduan Singkat Penggunaan</h4>
        <ol>
            <li>Pilih menu <b>Klasifikasi Berita</b> pada sidebar</li>
            <li>Masukkan teks berita ke dalam kolom input</li>
            <li>Klik tombol <b>Prediksi Kategori</b></li>
            <li>Sistem akan menampilkan hasil klasifikasi dan confidence score</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# =====================
# HALAMAN KLASIFIKASI
# =====================
elif selected == "Klasifikasi Berita":
    st.markdown(
        "<h1 style='text-align:center;'>📰 Klasifikasi Kategori Berita</h1>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("### ✍️ Masukkan Teks Berita")

    text_input = st.text_area(
        label="Teks Berita",
        height=280,
        placeholder="Contoh: Pemerintah meresmikan proyek infrastruktur nasional...",
        label_visibility="collapsed"
    )

    if st.button("🔍 Prediksi Kategori", use_container_width=True):
        if text_input.strip() == "":
            st.warning("⚠️ Silakan masukkan teks berita terlebih dahulu.")
        else:
            with st.spinner("⏳ Memproses prediksi..."):
                inputs = tokenizer(
                    text_input,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )

                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_label = label_map[str(pred_idx)]
                confidence = probs[pred_idx] * 100

            st.success("✅ Prediksi Berhasil")

            c1, c2 = st.columns(2)
            c1.metric("📂 Kategori Berita", pred_label)
            c2.metric("🎯 Confidence", f"{confidence:.2f}%")

            if confidence >= 80:
                badge = "🟢 High Confidence"
            elif confidence >= 60:
                badge = "🟡 Medium Confidence"
            else:
                badge = "🔴 Low Confidence"

            st.markdown(f"### {badge}")
            st.progress(int(confidence))

            st.subheader("📊 Probabilitas Setiap Kategori")

            df_prob = pd.DataFrame({
                "Kategori": [label_map[str(i)] for i in range(len(probs))],
                "Probabilitas (%)": probs * 100
            })

            fig, ax = plt.subplots()
            ax.barh(df_prob["Kategori"], df_prob["Probabilitas (%)"])
            ax.set_xlabel("Probabilitas (%)")
            ax.invert_yaxis()

            st.pyplot(fig)
            plt.close(fig)
