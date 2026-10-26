# RetrievalLab — Metrik Kılavuzu (her sayı nasıl hesaplanır)

Bu belge, Metrics dashboard'daki ve dışa aktarılan Excel/CSV'deki her metriği
tanımlar, formülünü verir, hesaplayan kodu (`app/metrics.py`) gösterir ve her
biri için somut bir örnek işler. Tüm sayılar dashboard'da, açılır (drill-down)
popup'ta, CSV'de ve Excel'de aynı şekilde hesaplanır — hepsi tek bir kod yolunu
paylaşır, bu yüzden asla çelişmezler.

---

## 0. Neyi ölçüyoruz

İki retrieval pipeline'ı aynı soruyu yanıtlar ve bağımsız olarak puanlanır:

- **A — Vektör RAG**: soruyu gömer (embedding), Pinecone'dan en iyi *k* parçayı
  (top-*k* chunk) getirir ve onlardan yanıtlar.
- **B — Akıl yürüten ajan (reasoning)**: bir doküman ağacında gezinir ve akıl
  yürüterek sayfaları *okur* (embedding yok).

**Değerlendirme birimi (doküman, sayfa) çiftidir.** Her soru bir altın etiket
`gold_pages` taşır — yanıtı gerçekten içeren sayfalar, ör.:

```json
"gold_pages": { "DIN_EN_ISO_1461_2022-12-00_DE_3388189.pdf": [23] }
```

Her iki pipeline da bir **kanıt kümesi** (evidence set) üretir; bu, altın kümeyle
karşılaştırılır:

- **A** için kanıt, getirilen top-*k* parçanın sayfalarıdır;
- **B** için kanıt, ajanın fiilen okuduğu sayfalardır.

**Sayfa düzeyinde, tekilleştirilmiş.** Getirilen parçalar, puanlamadan önce
benzersiz sayfalarına indirgenir; çünkü altın etiket sayfa biriminde tanımlıdır
(`app/metrics.py` içindeki `pageset()`). Üç parça da 23. sayfadan geliyorsa,
23. sayfa bir kez sayılır.

Boyunca şu tanımları kullanalım:
- **G** = altın sayfalar kümesi,
- **H** = kanıt sayfaları kümesi ("isabet" kümesi),
- **H ∩ G** = bulunan altın sayfalar.

---

## 1. Recall@k / Context Recall (Duyarlılık)

**Yanıtladığı soru:** *altın sayfaların kaçı bulundu?*

**Formül**
```
Recall = |H ∩ G| / |G|
```
Kod: `_recall(hit, gold)`. `G` boşsa `None` döner (cevaplanamaz kontrol sorusu —
ortalamalardan dışlanır, bkz. §8).

**A** için bu *Recall@k*'dır (top-*k* getirimle duyarlılık). Sabit *k*'sı olmayan
**B** için aynı formül, okuduğu sayfalar üzerinden *Context Recall*'dur.

**İşlenmiş örnek**
Altın `G = {(ISO1461, 23)}`. A'nın top-5 parçası şu sayfalara düşüyor:
`23, 23, 24, 5, 9` → benzersiz kanıt `H = {23, 24, 5, 9}`.
```
H ∩ G = {23}  →  Recall = 1 / 1 = 1,00 = %100
```
Altın iki sayfa olsaydı `G = {23, 24}` ve yalnızca 23 getirilseydi,
Recall = 1/2 = %50 olurdu.

---

## 2. Precision@k / Context Precision (Kesinlik)

**Yanıtladığı soru:** *getirilen sayfaların kaçı altındı?*

**Formül**
```
Precision = |H ∩ G| / |H|
```
Kod: `_precision(hit, gold)`. `H` boşsa: `G` de boşsa `None`, aksi halde `0.0`
(cevaplanabilir bir soruda hiçbir şey getirmemek 0 puandır).

**Önemli — payda, benzersiz getirilen SAYFA sayısıdır, k değil.** Üç parçayla
getirilen bir sayfa bir kez sayılır. Yani bu, sayfa düzeyinde *context
precision*'dır; "k parçanın kaçı ilgili" değildir.

**İşlenmiş örnek** (§1'deki getirimle aynı)
`H = {23, 24, 5, 9}` (4 benzersiz sayfa), `G = {23}`.
```
H ∩ G = {23}  →  Precision = 1 / 4 = 0,25 = %25
```
Dikkat: bu **%25**, %20 değil (= 1/5 chunk düzeyi); çünkü beş parça yalnızca dört
farklı sayfayı kapsıyor.

---

## 3. F1 (retrieval)

**Yanıtladığı soru:** *kesinlik ve duyarlılığı birleştiren tek dengeli skor.*

**Formül**
```
F1 = 2 · P · R / (P + R)      (eşdeğeri: 2·|H ∩ G| / (|H| + |G|))
```
Kod: `_f1(hit, gold)`. `G` boşsa `None`; ilgili hiçbir şey getirilmediyse
(`H ∩ G = ∅`) `0.0` döner.

**İşlenmiş örnek** (§1–2 ile aynı)
`P = 0,25`, `R = 1,00`.
```
F1 = 2 · 0,25 · 1,00 / (0,25 + 1,00) = 0,5 / 1,25 = 0,40 = %40
```
Doğrudan formülle kontrol: `2·|H∩G| / (|H|+|G|) = 2·1 / (4+1) = 2/5 = 0,40`. ✓

**İkinci örnek** — altın iki sayfa, bir fazladan getirim:
`G = {p1, p2}`, `H = {p1, p2, p3}` → `P = 2/3`, `R = 1` →
`F1 = 2·(2/3)·1 / (2/3 + 1) = 0,8 = %80`.

---

## 4. MRR — Ortalama Karşılıklı Sıra (yalnızca Pipeline A)

**Yanıtladığı soru:** *ilk altın sayfa, sıralı listede ne kadar yukarıda?*

**Formül (soru başına)**
```
RR = 1 / (ilk getirilen altın sayfanın sırası)      (altın getirilmediyse 0)
```
Kod: `_mrr_from_ranked(retrieved, gold)`. A'nın sıralı parça listesini (Pinecone
benzerlik sırasında) **sayfaya göre tekilleştirerek** dolaşır (her farklı sayfa
bir sonraki sırayı alır) ve ilk altın sayfada `1/sıra` döndürür. `G` boşsa
`None`. "MRR", bu RR değerlerinin sorular üzerindeki ortalamasıdır (§8).

Yalnızca A'nın bir sıralaması vardır; B bunun yerine *reads→gold* kullanır (§5).

**İşlenmiş örnek**
Sıralı sayfalar `[5, 5, 9, 23, 24]`, altın `G = {23}`. Sayfa tekilleştirmesinden
sonra sıralar 5→1, 9→2, 23→3.
```
ilk altın (23) 3. sırada  →  RR = 1/3 = 0,33
```
23 en başta getirilseydi RR = 1/1 = 1,00 olurdu. Hiç altın sayfa getirilmeseydi
RR = 0 olurdu.

---

## 5. reads→gold (yalnızca Pipeline B)

**Yanıtladığı soru:** *ajan ilk altın sayfaya ulaşana dek kaç sayfa okudu?*

**Formül (soru başına)**
```
reads→gold = ilk altın sayfanın B'nin okuma sırasındaki 1-tabanlı konumu
```
Kod: `_reads_to_gold(read_order, gold)`. `G` boşsa **ya da ajan hiç altın sayfa
okumadıysa** `None` döner — yani bu metrik yalnızca *başarılı* gezinmeleri
yansıtır ve B'nin duyarlılığıyla (kaçırmaları o yakalar) birlikte okunmalıdır.

**İşlenmiş örnek**
Ajan sayfaları şu sırayla okur: `[10, 23, 24]`, altın `G = {23}`.
```
ilk altın (23) okunan 2. sayfa  →  reads→gold = 2
```
Düşük olması iyidir (daha erken buldu). Ajan `[10, 24, 30]` okuyup 23'e hiç
ulaşmasaydı değer `None` olur ve koşu reads→gold ortalamasından dışlanır (ama
Recall'ı 0'dır).

---

## 6. Overlap — Jaccard(A getirdi, B okudu)

**Yanıtladığı soru:** *iki paradigma aynı sayfalara mı bakıyor?* (betimleyici, bir
kalite skoru değil).

**Formül**
```
Overlap = |A_sayfalar ∩ B_sayfalar| / |A_sayfalar ∪ B_sayfalar|
```
Kod: `_jaccard(a_set, b_set)`; `a_set` = A'nın getirdiği sayfalar, `b_set` =
B'nin okuduğu sayfalar. İkisi de boşsa `None`.

**İşlenmiş örnek**
A getirdi `{5, 9, 23, 24}`, B okudu `{23, 24, 30}`.
```
∩ = {23, 24} (2)   ∪ = {5, 9, 23, 24, 30} (5)   →  Overlap = 2/5 = 0,40
```
`0` = pipeline'lar ayrık sayfalara baktı; `1` = aynı sayfa kümeleri.

---

## 7. Judge — LLM-yargıç (yanıt kalitesi)

**Yanıtladığı soru:** *hangi pipeline daha iyi yanıt verdi?* Bu, getirimden
ayrıdır: §1–5 doğru sayfaların bulunup bulunmadığını ölçer; yargıç ise yazılı
yanıtın kalitesini ölçer.

**Yöntem** (`app/judge.py`):
- **Kör** — yargıç "Yanıt 1" / "Yanıt 2" görür, asla "vektör"/"reasoning" değil.
- **Rastgele sıra** — hangi pipeline'ın Yanıt 1 olacağına tohumlu (seeded) bir
  yazı-tura karar verir; konum yanlılığı böylece sönümlenir, seed verilirse
  tekrarlanabilir.
- **Farklı aile önerilir** — üretici GPT olduğundan, Anthropic bir yargıç
  öz-tercih yanlılığını en iyi azaltır (yapılandırılabilir; yoksa OpenAI
  yargıcına düşer).
- Her yanıt 1–5 arası **grounded (dayanaklı)**, **complete (eksiksiz)**,
  **correct (doğru)** puanlanır; yargıç bir kazanan döndürür: `1` / `2` / `tie`.

**Toplama:** dashboard `1→A`, `2→B` eşler ve kazanma paylarını raporlar:
yargılanan koşular üzerinden `A_win%`, `B_win%`, `tie%`.

**İşlenmiş örnek**
9 yargılanan soru: A 5 kez, B 3 kez kazanır, 1 berabere →
`A %56 · B %33 · tie %11` (5/9, 3/9, 1/9).

**Uyarı (tezde belirt):** A `[n]` ile, B `(Doküman, s.X)` ile atıf verir; bu
biçemsel ipuçları pipeline kimliğini sızdırabilir. Kör etiketler + rastgele sıra
birincil kontrollerdir. Aynı aileden yargıç (ör. GPT'nin GPT'yi yargılaması)
artık bir öz-tercih riski taşır.

---

## 8. Toplama — soru başına değerler nasıl tek bir hücre olur

Yukarıdaki her metrik önce **koşu başına** hesaplanır, sonra ortalanır. Bir koşu
grubu için (soru tipi, içerik tipi veya genel):

**Ortalama (koşular üzerinde makro-ortalama)**
```
mean = (1/N) · Σ xᵢ
```
Bu, soru başına değerlerin ortalamasıdır — `F1(ortalama P, ortalama R)` *değil*.
Kod: `_stats()`; alan, `_agg_system()` içinde koşular üzerinde toplanır.

**Standart sapma (popülasyon, ÷N)**
```
std = sqrt( (1/N) · Σ (xᵢ − mean)² )
```
Dikkat: **N**'e bölünür, N−1'e değil (popülasyon std, örneklem std değil).

**Gösterim: `mean% ± std`** — dashboard ve Excel "Summary" sayfası, Recall /
Precision / F1 için ortalama ile std'yi yüzde olarak gösterir (`mMean()`), ör.
`%11 ± 31`. **± bir güven aralığı değil, değerlerin yayılımıdır** — %95 güven
aralığı diye etiketleme.

**Altın metrikler için yalnızca altın taşıyan koşular sayılır.** Recall,
Precision, F1, MRR ve reads→gold **yalnızca altını olan koşular** (`has_gold`)
üzerinde ortalanır; çünkü bunlar altın olmadan tanımsızdır. Cevaplanamaz kontrol
soruları bu ortalamalardan dışlanır (ve ayrıca "Unanswerable" olarak sayılır);
**yargıç yine de onlara uygulanır**, çünkü "daha iyi yanıt, kaynaklarda
olmadığını açıkça söyleyendir".

**İşlenmiş örnek — `%11 ± 31` nereden geliyor**
Bir soru tipinde 9 soru var. Birinde Pipeline A altın sayfayı buldu (metrik =
%100); diğer sekizinde bulamadı (metrik = %0).
```
değerler = [%100, 0, 0, 0, 0, 0, 0, 0, 0]
mean     = 1,0 / 9                          = 0,111  → %11
var      = [ (1−0,111)² + 8·(0−0,111)² ] / 9 = 0,0988
std      = √0,0988                          = 0,314  → 31
```
→ **`%11 ± 31`** olarak gösterilir. Ortalamaya göre büyük bir std, tek altın
sayfalı sorularda recall/F1 için normaldir: değer neredeyse iki-uçludur (0 ya da
1), sistem "kimini yakalar, kimini kaçırır" ve bu yayılımın kendisi bir bulgudur.

**Tekrarlar.** `repeat = 1` iken grubun koşuları onun farklı sorularıdır; yani
yayılım *sorular arasıdır*. `repeat ≥ 2` iken her soru birkaç kez koşar
(determinizm yok) ve yayılım, sorular-arası ve koşu-arası varyansı karıştırır.

---

## 9. Verimlilik metrikleri

Betimleyici maliyet/hız sütunları, her şey gibi ortalanır (§8):

| Metrik | Anlamı | Kaynak |
|---|---|---|
| **Latency** | yanıt başına duvar-saati saniyesi | `latency_ms` |
| **Tokens** | istem + tamamlama token'ları | `usage.in + usage.out` |
| **Cost** | `token × fiyat` (`.env`'de `COST_PER_1M_*` yoksa 0) | `config.token_cost()` |
| **Effort** | A: LLM çağrı sayısı · B: araç (tool) çağrı sayısı | `calls` / `tool_calls` |

---

## 10. İçerik tipi kırılımı (prose vs table)

Her soru bir `content_type` taşır (`prose` ya da `table`). Bu, log'a akar ve
açılır popup ile Excel "Detail" sayfasında görünür. **Prose (düz metin)** yanıtlı
sorularla **table (tablo)** yanıtlı soruların retrieval kalitesini
karşılaştırmanı sağlar — multi-vector tablo yönteminin işe yarayıp yaramadığının
doğrudan sınavı. (İstenirse içerik tipine özel bir özet satırı eklenebilir.)

---

## 11. Tanımların özeti

| Metrik | Formül | Kod | Ne zaman None |
|---|---|---|---|
| Recall@k / Context Recall | `\|H∩G\| / \|G\|` | `_recall` | altın yok |
| Precision@k / Context Precision | `\|H∩G\| / \|H\|` | `_precision` | altın yok **ve** hiçbir şey getirilmedi |
| F1 | `2·\|H∩G\| / (\|H\|+\|G\|)` | `_f1` | altın yok |
| MRR (A) | `1 / ilk altın sayfanın sırası` | `_mrr_from_ranked` | altın yok |
| reads→gold (B) | okunan ilk altın sayfanın konumu | `_reads_to_gold` | altın yok ya da hiç okunmadı |
| Overlap | `\|A∩B\| / \|A∪B\|` (sayfalar) | `_jaccard` | ikisi de boş |
| Judge | A/B/tie kazanma payı | `app/judge.py` | yargılanmadı |
| mean / std | makro-ortalama / popülasyon std | `_stats` | sayısal değer yok |

**Tezde belirtilmesi gereken temel metodolojik noktalar:** metrikler sayfa
düzeyindedir (parçalar sayfalara tekilleştirilir, bu yüzden precision'ın paydası
sayfadır, *k* değil); sorular üzerinde makro-ortalama alınır; popülasyon standart
sapması kullanılır; `±` yayılımdır, güven aralığı değildir; cevaplanamaz sorular
retrieval ortalamalarından dışlanır ve yargıç tarafından ele alınır.
