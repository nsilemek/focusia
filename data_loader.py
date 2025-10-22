# -*- coding: utf-8 -*-
"""
data_loader.py
--------------
- focus_tips.json içeriğini Chroma vektör veritabanına gömer.
- Eğer indeks daha önce oluşturulmuşsa (persist_dir doluysa) aynı indeksi yükler.
- Embedding: sentence-transformers/all-MiniLM-L6-v2
- Vektör DB: Chroma

Kullanım:
    from data_loader import build_or_load_chroma
    db = build_or_load_chroma()

Notlar:
- HF Spaces'te 50GB kotayı doldurmamak için indeksleri kalıcı dizin yerine
  /tmp altında tutmak gerekebilir. Bunu yapmak için ENV değişkeni tanımlayıp
  PERSIST_DIR yerine CHROMA_PERSIST_DIR kullanabilir (aşağıdaki yorumlara bakılabilir).
"""

from pathlib import Path
import json
from typing import List, Dict, Any

# LangChain v0.2+ import yolları
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- Varsayılan ayarlar (ENV ile özelleştirilebilir) ----
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # hızlı ve küçük, demo için ideal
PERSIST_DIR = "chroma_db"                              # kalıcı indeks klasörü (varsayılan)
JSON_PATH = "focus_tips.json"                          # veri kaynağı dosyası


def _normalize_record(row: Dict[str, Any]) -> Document | None:
    """
    JSON'daki tek bir satırı güvenli şekilde LangChain Document'e çevirir.
    - Boş/eksik içerikleri ve çok kısa kırpıntıları eler.
    - 'topic' alanını metadata'ya koyar, 'content' + 'topic' birleşimini page_content yapar.
    """
    if not isinstance(row, dict):
        return None

    topic = (row.get("topic") or "").strip()
    content = (row.get("content") or "").strip()
    if not content:
        return None

    text = f"{topic}\n\n{content}" if topic else content

    # Çok kısa (ör. 1-2 kelimelik) kırpıntıları atla; embedding kalitesini düşürmemek için.
    if len(text) < 8:
        return None

    return Document(page_content=text, metadata={"topic": topic})


def build_or_load_chroma(
    json_path: str = JSON_PATH,
    persist_dir: str = PERSIST_DIR,
    embedding_model: str = EMB_MODEL,
) -> Chroma:
    """
    focus_tips.json'dan Chroma DB'yi kurar ya da var olanı yükler.

    Parametreler:
        json_path: JSON veri dosyası yolu (varsayılan: focus_tips.json)
        persist_dir: Chroma indeksinin saklanacağı klasör (varsayılan: chroma_db)
                     # NOTE: HF Spaces kota sorunu yaşanıyorsa, (örn. 50GB limit)
                     # persist_dir'i ENV ile /tmp/chroma_db gibi geçici bir dizine
                     # yönlendirebilir: CHROMA_PERSIST_DIR=/tmp/chroma_db
        embedding_model: sentence-transformers model adı

    Döndürür:
        Chroma: Vektör veritabanı nesnesi (LangChain uyumlu)
    """
    # Kalıcı dizini hazırla (yoksa oluştur)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    # Embedding modeli yükleniyor
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Eğer indeks klasörü doluysa (daha önce oluşturulmuşsa), doğrudan yükle
    has_index = any(Path(persist_dir).glob("*"))
    if has_index:
        print(f"[Chroma] Loaded existing index from: {persist_dir}")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # İlk kurulum: JSON'dan dokümanları oku
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"{json_path} bulunamadı.")

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        # Beklenen format: [{"topic": "...", "content": "..."}, ...]
        raise ValueError(f"{json_path} formatı geçersiz: kök seviye bir liste bekleniyor.")

    # Kayıtları normalize et + yinelenenleri ayıkla (aynı topic + içerik kombinasyonunu tekilleştir)
    docs: List[Document] = []
    seen = set()
    for row in data:
        doc = _normalize_record(row)
        if not doc:
            continue
        key = (doc.metadata.get("topic", ""), doc.page_content.strip())
        if key in seen:
            continue
        seen.add(key)
        docs.append(doc)

    if not docs:
        raise ValueError(f"{json_path} içinde geçerli kayıt bulunamadı.")

    # Belge listesinden kalıcı Chroma indeksi oluştur
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"[Chroma] Built new index with {len(docs)} docs → {persist_dir}")
    return db

