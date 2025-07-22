# 📊 Samsung Competitor Product Mapping

> A structured reference of competitor products for Samsung’s product catalogue (e.g. in Mobile -> flagship, foldables, mid-range, budget) across various brands and price segments. Useful for search relevance, SEO strategy, and competitive analysis.

---

## 🧠 Objective

To identify and document **similar or substitute products from other brands** for each Samsung device across categories like phones, foldables, smartwatches, tablets, and earbuds — enabling structured comparison and powering downstream tasks.

---

## 🔍 Methodology

1. **Input:** A list of Samsung products (existing, upcoming, rumored).
2. **Category Detection:** Classify each Samsung product into a category (e.g., flagship smartphone, foldable, mid-range phone, etc.).
3. **Competitor Discovery:** Retrieve or map similar competitor devices that match on:

   * **Form factor** (e.g., flip, fold, bar phone)
   * **Target audience / use-case**
   * **Hardware or feature similarity**
   * **Market positioning** (flagship, mid-range, budget)
4. **Output Formats:** Structured into CSV and JSON.

---

## 📁 Output Files

| Format  | Description                                                                     | Download                                                           |
| ------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 📄 CSV  | Tabular data with columns: `Samsung Product`, `Category`, `Competitor Products` | [Download CSV]()   |
| 📄 JSON | Dictionary with product as key and details as values (category + competitors)   | [Download JSON]() |

---

## 🧾 Sample Output (CSV)

```csv
Samsung Product,Competitor Products
Galaxy S24,"iPhone 15, Google Pixel 8, OnePlus 12R, Xiaomi 14, ASUS Zenfone 10"
Galaxy S24+,"iPhone 15 Plus, OnePlus 12, Xiaomi 13 Pro, Vivo X90, Realme GT5"
Galaxy S24 Ultra,"iPhone 15 Pro Max, Xiaomi 14 Ultra, Vivo X100 Pro, Honor Magic6 Pro, Huawei Mate 60 Pro+"
Galaxy S24 FE,"iPhone SE 3, Pixel 7a, OnePlus Nord 3, Realme GT 6T, Xiaomi 13 Lite"
Galaxy F15 5G,"Redmi Note 13 5G, Realme Narzo 60x, iQOO Z9x, Infinix Zero 5G 2023, Lava Blaze 5G"
Galaxy A35 5G,"Redmi Note 13 Pro 5G, Realme Narzo 70 Pro 5G, iQOO Z9, Infinix GT 20 Pro, Moto G73 5G"
Galaxy A55 5G,"Nothing Phone (2a), Pixel 6a, Poco F5, Vivo V30, Xiaomi CIVI 3"
Galaxy M15 5G,"Redmi 13C 5G, Lava Storm 5G, Realme Narzo 60x, Infinix Smart 8 Plus, itel P55 5G"
Galaxy M55 5G,"iQOO Z9, Poco X6, Realme Narzo 70 Pro, Infinix GT 20 Pro, Redmi Note 13 Pro+"
Galaxy M35 5G,"Realme 12x 5G, Redmi Note 13 5G, iQOO Z9x, Infinix Zero 30 5G, Lava Blaze Curve 5G"
Galaxy A06,"Redmi A3, Realme C53, itel A70, Infinix Smart 8, Lava O2"
Galaxy A16,"Redmi 13C, Realme Narzo N53, Infinix Hot 30i, itel P40, Lava Yuva 3"

```
```json
{
  "Galaxy S24": [
    "iPhone 15",
    "Google Pixel 8",
    "OnePlus 12R",
    "Xiaomi 14",
    "ASUS Zenfone 10"
  ],
  "Galaxy S24+": [
    "iPhone 15 Plus",
    "OnePlus 12",
    "Xiaomi 13 Pro",
    "Vivo X90",
    "Realme GT5"
  ],
  "Galaxy S24 Ultra": [
    "iPhone 15 Pro Max",
    "Xiaomi 14 Ultra",
    "Vivo X100 Pro",
    "Honor Magic6 Pro",
    "Huawei Mate 60 Pro+"
  ],
  "Galaxy S24 FE": [
    "iPhone SE 3",
    "Pixel 7a",
    "OnePlus Nord 3",
    "Realme GT 6T",
    "Xiaomi 13 Lite"
  ],
  "Galaxy F15 5G": [
    "Redmi Note 13 5G",
    "Realme Narzo 60x",
    "iQOO Z9x",
    "Infinix Zero 5G 2023",
    "Lava Blaze 5G"
  ],
  "Galaxy A35 5G": [
    "Redmi Note 13 Pro 5G",
    "Realme Narzo 70 Pro 5G",
    "iQOO Z9",
    "Infinix GT 20 Pro",
    "Moto G73 5G"
  ],
  "Galaxy A55 5G": [
    "Nothing Phone (2a)",
    "Pixel 6a",
    "Poco F5",
    "Vivo V30",
    "Xiaomi CIVI 3"
  ],
  "Galaxy M15 5G": [
    "Redmi 13C 5G",
    "Lava Storm 5G",
    "Realme Narzo 60x",
    "Infinix Smart 8 Plus",
    "itel P55 5G"
  ],
  "Galaxy M55 5G": [
    "iQOO Z9",
    "Poco X6",
    "Realme Narzo 70 Pro",
    "Infinix GT 20 Pro",
    "Redmi Note 13 Pro+"
  ],
  "Galaxy M35 5G": [
    "Realme 12x 5G",
    "Redmi Note 13 5G",
    "iQOO Z9x",
    "Infinix Zero 30 5G",
    "Lava Blaze Curve 5G"
  ],
  "Galaxy A06": [
    "Redmi A3",
    "Realme C53",
    "itel A70",
    "Infinix Smart 8",
    "Lava O2"
  ],
  "Galaxy A16": [
    "Redmi 13C",
    "Realme Narzo N53",
    "Infinix Hot 30i",
    "itel P40",
    "Lava Yuva 3"
  ]
}
```
---

## 🛠️ Applications

* **Search & Recommendation Systems:** Boost relevance for alternative product queries.
* **SEO & Paid Campaigns:** Optimize content for cross-brand comparison search.
* **Product Benchmarking:** Competitive analysis for internal marketing & pricing.
* **Customer Service Bots:** Enhance fallback responses in case of OOS/no-results queries.

---

## 📌 Notes

* Products marked as “Upcoming” or “Rumored” are mapped based on expected segment and leaked specs.
* Price overlap is considered but not mandatory; **functional and intent similarity** is prioritized.

---

📥 For automation or bulk usage, please integrate the JSON version into your backend pipelines or export to knowledge bases.

---
