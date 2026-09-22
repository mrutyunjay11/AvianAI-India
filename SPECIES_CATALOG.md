# AvianAI India - Recognized Indian Species Catalog

**System:** `AvianNet-v3` | **Total Global Classes:** `11,560` | **Indian Subcontinent Species:** `1,074`

This catalog contains the primary bird species of the Indian subcontinent recognized by the model with geographic ecozone mapping, including localized species distribution for **Rajkot & Saurashtra (Gujarat)**.

---

## Geographic Ecozone Breakdown

| Ecozone Region | Region Code | Species Count | Primary Habitats & States |
|---|---|---|---|
| **Pan-India Unified** | `pan-india` | **1,074** | Complete Indian Subcontinent |
| **South India & Sri Lanka** | `south-asia-peninsular` | **644** | Western Ghats, Deccan Plateau, Kerala, Tamil Nadu, Karnataka, Sri Lanka |
| **Indo-Gangetic Plain** | `indo-gangetic` | **814** | Northern Plains, Punjab, UP, Bihar, West Bengal, Assam |
| **Himalayan Mountain Zone** | `himalaya` | **812** | Jammu & Kashmir, Ladakh, Himachal, Uttarakhand, Sikkim, Arunachal |
| **Global Catalog** | `all` | **11,560** | Worldwide avian bioacoustic taxonomy |

---

## Local Avifauna of Rajkot & Saurashtra (Gujarat)

GPS Coordinates: `22.30° N, 70.80° E`

The Saurashtra peninsula and Rajkot district feature diverse habitats ranging from semi-arid scrublands, agricultural plains, to freshwater reservoirs (Nyari Dam, Aji Dam, Lalpari Lake, and proximity to Khijadiya & Nal Sarovar). 

Below are 5 iconic local bird species that thrive in the Rajkot region and are accurately detected by the system:

### 1. Indian Peafowl (*Pavo cristatus*)
- **Local Gujarati Name:** મોર (Mor) / ઢેલ (Dhel)
- **Model Global Index:** `7832`
- **Habitats in Rajkot:** Aji riverbanks, Nyari catchment, temple groves, agricultural belts around Rajkot, Gondal, and Morbi.
- **Acoustic Characteristics:** Loud, echoing "may-awe", resonant trumpet-like territorial calls audible across 1 km.

### 2. Indian Roller (*Coracias benghalensis*)
- **Local Gujarati Name:** નીલકંઠ (Nilkanth) / ચાષ (Chash)
- **Model Global Index:** `2840`
- **Habitats in Rajkot:** Open agricultural fields, roadside telephone wires, scrublands along Rajkot-Jamnagar highway.
- **Acoustic Characteristics:** Harsh, grating "krak-krak" or "chack" calls during display flights.

### 3. Common Kingfisher (*Alcedo atthis*) & White-throated Kingfisher (*Halcyon smyrnensis*)
- **Local Gujarati Name:** કલકલીયો (Kalkaliyo) / કીકીચીયો (Kikichiyo)
- **Model Global Index:** `294` / `4865`
- **Habitats in Rajkot:** Nyari Dam reservoir, Aji-1 and Aji-2 dams, Lalpari Lake, Randarda Lake.
- **Acoustic Characteristics:** High-pitched, piercing "chee-kee" whistles while perching or hunting over water bodies.

### 4. Red-vented Bulbul (*Pycnonotus cafer*)
- **Local Gujarati Name:** બુલબુલ (Bulbul)
- **Model Global Index:** `9294`
- **Habitats in Rajkot:** City gardens, Jubilee Garden, Race Course ring road trees, terrace gardens, and orchard farms.
- **Acoustic Characteristics:** Cheerful, chattering multi-syllable whistling "be-care-ful" phrases.

### 5. Oriental Magpie-Robin (*Copsychus saularis*)
- **Local Gujarati Name:** ડૈયર (Daiyer)
- **Model Global Index:** `2836`
- **Habitats in Rajkot:** Urban parks, shaded residential compounds, University campus gardens.
- **Acoustic Characteristics:** Rich, melodious, variable whistles delivered from high perches at dawn and dusk.

---

## Other Notable Birds of Saurashtra / Gujarat in the Catalog

| # | Common Name | Gujarati Name | Scientific Name | Global Index | Primary Habitats in Gujarat |
|---|---|---|---|---|---|
| 6 | **Asian Koel** | કોયલ (Koyal) | *Eudynamys scolopaceus* | `4071` | Gardens, neem & banyan trees across Rajkot |
| 7 | **Greater Flamingo** | હંસ / મોટો સુરખાબ (Surkhab) | *Phoenicopterus roseus* | `8131` | Nal Sarovar, Gulf of Kutch, coastal mudflats |
| 8 | **Lesser Flamingo** | નાનો સુરખાબ (Nano Surkhab) | *Phoeniconaias minor* | `8127` | Salt pans, Khijadiya wetlands, Little Rann of Kutch |
| 9 | **Black Drongo** | કાળો કોશી (Kalo Koshi) | *Dicrurus macrocercus* | `3575` | Farmland poles, scrublands around Saurashtra |
| 10 | **Grey Francolin** | દેશી તેતર (Deshi Tetar) | *Ortygornis pondicerianus* | `7530` | Dry scrublands, open grasslands of Rajkot & Surendranagar |
| 11 | **Indian Silverbill** | પાવડારી / સફેદ મુનિયા (Pavadari) | *Euodice malabarica* | `4105` | Grasslands and dry acacia scrub around Saurashtra |
| 12 | **Brahminy Kite** | બ્રાહ્મણી ગરુડ (Brahmani Garud) | *Haliastur indus* | `4875` | Coastal rivers, Saurashtra coast & large wetlands |
| 13 | **Coppersmith Barbet** | ટુકટુકીયો (Tuktukiyo) | *Psilopogon haemacephalus* | `6224` | Fruiting fig trees in urban Rajkot |
| 14 | **Common Myna** | કાબર (Kabar) | *Acridotheres tristis* | `83` | Ubiquitous across town streets and agricultural lands |
| 15 | **Spotted Dove** | હોલો / છાપિયો હોલો (Chhapiyo Holo) | *Spilopelia chinensis* | `10121` | Scrublands, suburban gardens, farmlands |

---

## Dynamic Geographic Filtering for Rajkot

To target birds of the Rajkot & Saurashtra ecozone with precision coordinates:
```python
from inference.model import get_model

classifier = get_model()

# Run prediction with Rajkot GPS coordinates (22.30° N, 70.80° E)
result = classifier.predict(
    "samples/indian_birds/01_indian_peafowl.wav",
    lat=22.3039,
    lon=70.8022,
    region="pan-india",
    threshold=0.25,
    top_k=5,
)

print(result.format_summary())
```
Non-occurring species in the target ecozone have their probabilities zeroed out before computing the final rankings, preventing false positives from distant continents.
