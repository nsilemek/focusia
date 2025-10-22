#  Focusia – Yapay Zeka Destekli Odak Koçu 
<img width="500" height="500" alt="logo" src="https://github.com/user-attachments/assets/9fe0bf6e-4cf5-40ad-9c17-67e2b57b5057" />


> **Focusia**, modern çağın dikkat dağınıklığı sorununa çözüm sunmak için geliştirilen  
> **RAG (Retrieval-Augmented Generation)** tabanlı kişisel bir **odaklanma koçudur**.  
> Türkçe olarak senin durumuna uygun öneriler üretir, kısa egzersizlerle zihnini yeniden odaklamanı sağlar.  

---

##  Canlı Demo

Uygulamayı hemen çevrimiçi olarak deneyebilirsin:  
👉 **[Focusia AI Focus Mentor – Hugging Face Spaces](https://huggingface.co/spaces/lisidne6/aifocus-mentor)**

> Model: `Qwen/Qwen2.5-7B-Instruct`  
> Ortam: `Hugging Face Inference API`  
> Arayüz: `Streamlit`

---

## 🎬 Demo Görseli


![focusai-1](https://github.com/user-attachments/assets/0efd8df4-1fc4-4c8e-b852-742290244bbe)

##  Özellikler

● Kişisel Türkçe öneriler – her kullanıcı girdisine göre benzersiz yanıt  
● RAG tabanlı yapı – focus_tips.json verilerini dinamik olarak kullanır  
● MMR (Maximal Marginal Relevance) arama algoritması – tekrarlayan içerikleri azaltır  
● Minimalist, gradient temalı arayüz (lacivert → mor geçişli)  
● Expander içi kaynak görünümü – kullanıcı isterse model dayanaklarını inceleyebilir  
● Tamamen açık kaynak – kolayca genişletilebilir ve özelleştirilebilir  

##  Mimarinin Özeti
```
Kullanıcı → Streamlit Arayüzü (app.py)  
          ↓  
RAG Pipeline (rag_pipeline.py)  
          ↓  
Retriever (Chroma + SentenceTransformer)  
          ↓  
LLM (Qwen 2.5-7B-Instruct via Hugging Face Inference API)
```


**Chroma DB:** *JSON veri setini vektör uzayına dönüştürür.*

**Retriever:** *Kullanıcı girdisiyle en alakalı kayıtları getirir.*

**LLM:** *Bağlamı ve kullanıcı mesajını birlikte analiz edip anlamlı ve Türkçe öneriler üretir.*

## Dosya Yapısı
```
focusia/  
├─ app.py                → Ana Streamlit uygulaması (UI + RAG entegrasyonu)  
├─ rag_pipeline.py       → LLM adaptörü ve prompt zinciri  
├─ data_loader.py        → Chroma veritabanını kurar/yükler  
├─ focus_tips.json       → Odak önerileri veri seti  
├─ requirements.txt      → Python bağımlılıkları  
├─ logo.png              → Uygulama logosu  
├─ .streamlit/config.toml → Tema ayarları  
└─ README.md             → Proje dokümantasyonu
```

##  Arayüz Teması

Arka plan	Lacivert → mor geçişli gradient  
Form alanı	Koyu lacivert, yuvarlatılmış kenarlıklar  
Butonlar	Mor tonlu gradient, hover animasyonu  
Öneri kartı	Şeffaf, gölgeli blur efekti  
Expander	“📚 Kaynakları Gör” – RAG referanslarını içerir  

## Kullanılan Teknolojiler

| **Katman**        | **Teknoloji** |
|--------------------|---------------|
|  Arayüz          | Streamlit |
|  LLM             | Hugging Face Inference API (`Qwen2.5-7B-Instruct`) |
|  Vektör Veritabanı | Chroma |
|  Embeddings       | Sentence Transformers (`all-MiniLM-L6-v2`) |
|  RAG Pipeline     | LangChain Core + özel retrieval adaptörü |
|  Stil / UI        | CSS gradient, shadow ve blur efektleri |


## Örnek Kullanım

> **Kullanıcı:**  
> “Sürekli telefona bakıyorum, başladığım işi bitiremiyorum.”

** Model Yanıtı:**

> Bildirimleri sessize al _(telefonu başka odada tut)_
> 25 dakikalık **Pomodoro döngüleri** oluştur  
> Masanı sadeleştir, dikkat dağıtan nesneleri kaldır  

> **Mini Egzersiz:**  
> 2 dakika boyunca yalnızca nefesine odaklan. Her nefeste 4’e kadar say.

---

<div align="center">

✨ **Developed with 💜 by [Nilüfer Silemek](www.linkedin.com/in/nilufersilemek)** ✨  

</div>
