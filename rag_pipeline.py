# -*- coding: utf-8 -*-
"""
rag_pipeline.py
---------------
Focusia'nın LLM ve RAG (Retrieval-Augmented Generation) akışını kurar.

- HF Inference API ile sohbet (chat_completion) öncelikli; destek yoksa text_generation'a düşer.
- Çıktı, _postprocess ile sadeleştirilir: madde işaretleri, kısa açıklamalar ve mini egzersiz.
- _SimpleRAGAdapter sınıfı, LangChain v0.2+ ile gelen retriever API değişikliklerine
  uyumludur (hem .get_relevant_documents hem de .invoke yollarını destekler).
"""

import os
from typing import Any, List, Optional, Dict
from huggingface_hub import InferenceClient

# LangChain v0.2+ çekirdek importları (chains modülüne gerek yok)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

# Sistem rolü: modelin üslubunu ve çıktıyı sabitlemek için.
SYSTEM_TR = (
    "Sen bir odaklanma koçusun. Türkçe, doğal ve kısa cümlelerle konuş.\n"
    "- 'Sen' diye hitap et.\n"
    "- Üç net öneri ve bir mini egzersiz yaz.\n"
    "- Yapay/çeviri kokan ifadeler kullanma; sohbet tonu koru.\n"
    "- İngilizce kelime, emoji, [INST] gibi kalıntılar yazma.\n"
)

def _postprocess(text: str) -> str:
    """
    Model çıktısını düzenler:
    - Sistem/şablon kalıntılarını temizler (örn. [INST], "Yanıt:")
    - Noktalama ve madde işaretlerini normalize eder
    - Kısa açıklama satırlarını (varsa) bir önceki maddeye parantez içinde ekler
    - Sonuç yoksa güvenli geri dönüş yapar
    - Mini egzersiz yoksa ekler
    """
    if not text:
        return ""

    # Fazla etiket ve semboller
    for bad in ("[/INST]", "[INST]", "Yanıt:", "Yanıtı:", "Answer:", "Response:"):
        text = text.replace(bad, "")
    text = text.replace("•", "-").replace("●", "-").replace("►", "-")

    # Satır bazlı işleme
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fixed: List[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]

        # "Kaynak" / "Source" başlığından sonrasını alma (RAG dayanaklarını UI'da gösteriyoruz)
        if ln.lower().startswith(("kaynak", "source")):
            break

        # Başındaki numara/madde karakterlerini temizle (örn. "1.", "2)", "-")
        while ln and (ln[0].isdigit() or ln[0] in "-•.–"):
            ln = ln[1:].lstrip(" .-)")

        ln = ln.strip()
        if not ln:
            i += 1
            continue

        # Sonraki satır kısa bir açıklamaysa paranteze al (10 kelime ve altı)
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if (
                nxt
                and not nxt[0].isdigit()
                and not nxt.startswith("-")
                and len(nxt.split()) <= 10
                and not nxt.lower().startswith(("kaynak", "source"))
            ):
                ln = f"{ln} ({nxt})"
                i += 1  # açıklama satırını tükettik

        fixed.append(ln)
        i += 1

    # Madde listesine dönüştür
    out = ("- " + "\n- ".join(fixed)) if fixed else text

    # Mini egzersiz ekle (yoksa)
    if "egzersiz" not in out.lower():
        out += "\n\nMini egzersiz: 2 dakika nefesine odaklan; her nefeste 4'e kadar say."

    return out.strip()


class HFClientLLM(LLM):
    """
    Hugging Face Inference API adaptörü.
    - Önce chat_completion kullanır (Qwen gibi sohbet-tabanlı modeller için ideal).
    - chat_completion başarısız olursa text_generation'a geri düşer.
    - LangChain LLM arayüzünü uygular: .invoke / _call üzerinden string alır, string döner.
    """
    client: InferenceClient
    max_new_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.8
    repetition_penalty: float = 1.08

    @property
    def _llm_type(self) -> str:
        return "hf_inferenceclient"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Tek bir string prompt alır ve modelden yanıt üretir.
        stop: (opsiyonel) durdurma dizeleri içeriyorsa çıktıyı bu dizelerden önce keser.
        """
        try:
            # 1) Sohbet arayüzü (tercih edilen yol)
            out = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_TR},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            choice = out.choices[0]
            msg: Dict[str, Any] = getattr(choice, "message", choice.get("message", {}))  # SDK sürüm uyumu
            text = getattr(msg, "content", None) or msg.get("content") or str(out)
        except Exception:
            # 2) text_generation'a düş (bazı modellerde yalnızca bu desteklenir)
            out = self.client.text_generation(
                prompt=f"{SYSTEM_TR}\n\nKullanıcı mesajı:\n{prompt}\n\nYanıt:",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
            )
            text = out if isinstance(out, str) else str(out)

        # Çıktıyı sadeleştir
        text = _postprocess(text)

        # stop dizeleri varsa kes
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text


def make_llm() -> HFClientLLM:
    """
    HF Inference Client + LLM adaptörünü hazırlar.
    - Token'ı HF_TOKEN / HUGGINGFACEHUB_API_TOKEN / HF_API_TOKEN değişkenlerinden alır.
    - Model ID'yi HF_MODEL'den okur; yoksa Qwen/Qwen2.5-7B-Instruct kullanır.
    """
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or ""
    )
    if not token:
        raise RuntimeError("HF token yok. HF_TOKEN (veya HUGGINGFACEHUB_API_TOKEN/HF_API_TOKEN) ekleyin.")

    model_id = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    client = InferenceClient(model=model_id, token=token)
    print(f"[HF] Using model: {model_id}")
    return HFClientLLM(client=client)


# ---- Basit, bağımsız RAG zinciri (chains modülüne ihtiyaç yok) ----

def _format_docs(docs: List) -> str:
    """Retriever'dan gelen belge listesini tek bir metin bağlama dönüştürür."""
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


class _SimpleRAGAdapter:
    """
    app.py ile arayüz uyumlu bir mini RAG zinciri.
    - .invoke({'query': ...}) çağrısını destekler
    - retriever: LangChain VectorStoreRetriever veya benzeri bir arayüz
      (hem .get_relevant_documents hem de .invoke çağrıları denenir)
    - llm: HFClientLLM (string alır, string döner)
    - prompt: PromptTemplate (değişkenler: {context}, {input}/{question})
    """
    def __init__(self, retriever, llm: LLM, prompt: PromptTemplate):
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    def _retrieve(self, q: str) -> List:
        """
        Arama katmanı:
        - Eski API: retriever.get_relevant_documents(q)
        - Yeni API: retriever.invoke(q) veya retriever.invoke({"query": q})
        Dönüş: belge listesi
        """
        # 1) Eski API: doğrudan belge listesi döner
        if hasattr(self.retriever, "get_relevant_documents"):
            return self.retriever.get_relevant_documents(q)

        # 2) Yeni API: invoke (bazı sürümler string, bazıları dict ister)
        try:
            res = self.retriever.invoke(q)
        except Exception:
            res = self.retriever.invoke({"query": q})

        if res is None:
            return []
        if isinstance(res, list):
            return res
        return [res]  # tek dökümanı listeye sar

    def invoke(self, payload: dict):
        """
        Dış arayüz:
        payload: {'query': '...'} veya {'input': '...'}
        Dönüş: {'result': <model çıktısı>, 'source_documents': <bağlam belgeleri>}
        """
        q = payload.get("query") or payload.get("input") or ""
        docs = self._retrieve(q)
        context = _format_docs(docs)

        # Prompt'u doldur ve LLM'yi çağır
        filled = self.prompt.format(context=context, input=q, question=q)
        answer = self.llm.invoke(filled)

        return {"result": answer, "source_documents": docs}


def make_retrieval_chain(retriever):
    """
    RAG zincirini kurar:
    - Türkçe, kısa ve uygulanabilir öneri üretimine odaklı bir prompt
    - HFClientLLM ile yanıt üretimi
    - _SimpleRAGAdapter ile uyumlu .invoke arayüzü
    """
    # Prompt değişkenleri: {context} ve {input}/{question}
    prompt = PromptTemplate.from_template(
        "Aşağıdaki kullanıcı mesajına, verilen bağlama dayanarak kısa ve uygulanabilir öneriler üret.\n"
        "- Tamamen Türkçe yaz.\n"
        "- 3 madde + 1 mini egzersiz ver.\n"
        "- Gereksiz laf kalabalığından kaçın.\n\n"
        "Bağlam:\n{context}\n\nKullanıcı: {input}\n\nYanıt:"
    )
    llm = make_llm()
    return _SimpleRAGAdapter(retriever, llm, prompt)

