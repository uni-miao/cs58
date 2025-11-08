### Data file description

* `altmetric_mentions_original_labeled.csv`：Contains all Original Mentions。
* `altmetric_mentions_retraction_labeled.csv`：Contains all Retraction Mentions。

The last column in the data set is the label column **"label"**. Please select the required label according to your needs.

---

### Label classification method

#### 1. Simply divide by **time before and after retraction**

| File type | Before Retraction | After Retraction |
| :--- | :--- | :--- |
| **Retraction Mentions** | `R_BEFORE_COMENTION` + `R_BEFORE_EXCL_NORM` + `R_BEFORE_EXCL_ABNORM` | `R_AFTER` |
| **Original Mentions** | `O_BEFORE` | `O_AFTER_COMENTION` + `O_AFTER_EXCL_NORM` + `O_AFTER_EXCL_ABNORM` |

#### 2. Divided by **scientific information dissemination time series structure**



**=== Original labels ===**

| label | count | share_of_total_% | count_classified | share_of_classified_% |
| :--- | :--- | :--- | :--- | :--- |
| `O_AFTER_COMENTION` | 8078 | 27.551160 | 8078.0 | 27.866703 |
| `O_AFTER_EXCL_ABNORM` | 3718 | 12.680764 | 3718.0 | 12.825997 |
| `O_AFTER_EXCL_NORM` | 2660 | 9.072306 | 2660.0 | 9.176211 |
| `O_BEFORE` | 14532 | 49.563438 | 14532.0 | 50.131089 |
| `UNCLASSIFIED` | 332 | 1.132333 | NaN | 0.000000 |

**=== Retraction labels ===**

| label | count | share_of_total_% | count_classified | share_of_classified_% |
| :--- | :--- | :--- | :--- | :--- |
| `R_AFTER` | 12729 | 82.276517 | 12729.0 | 84.905283 |
| `R_BEFORE_COMENTION` | 1902 | 12.293969 | 1902.0 | 12.686766 |
| `R_BEFORE_EXCL_ABNORM` | 66 | 0.426605 | 66.0 | 0.440235 |
| `R_BEFORE_EXCL_NORM` | 295 | 1.906793 | 295.0 | 1.967716 |
| `UNCLASSIFIED` | 479 | 3.096115 | NaN | 0.000000 |

---

### Label Typical Meaning

#### Original Mentions

* **`O_BEFORE`**
    : News or blogs reporting the original paper before it was officially retracted.
* **`O_AFTER_COMENTION`**
    : The same article mentions both the original and the retraction.
* **`O_AFTER_EXCL_NORM`**
    : News continues to cite the original paper after retraction but includes a notice.
* **`O_AFTER_EXCL_ABNORM`**
    : Continues citing the original paper after retraction, without any correction note.

#### Retraction Mentions

* **`R_AFTER`**
    : Reports on the retraction after it officially occurred.
* **`R_BEFORE_COMENTION`**
    : A retraction is mentioned together with the original paper before the formal retraction date.
* **`R_BEFORE_EXCL_NORM`**
    : Reports a retraction before the official date, but the information is later verified.
* **`R_BEFORE_EXCL_ABNORM`**
    : Claims a retraction before it actually happens, without subsequent verification.
