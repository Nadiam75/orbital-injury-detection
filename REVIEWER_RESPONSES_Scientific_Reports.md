# Response to Reviewers — *Scientific Reports*

**Manuscript:** Orbital fracture detection and severe globe injury (SGI) prediction from CT  
**Prepared for:** Paste into Google Docs (headings below map to title / subtitle / section levels)

---

# Reviewer 1 — Comments and responses

---

## Question 1 — Figure captions

### *Reviewer comment*
I do not see the captions of each figure.

### **Response**
We thank the reviewer for raising this point. **Figure captions are already provided in the manuscript:** they are **compiled in the figure-caption section at the end of the manuscript file** (rather than repeated directly under each inline figure in the main PDF flow). If this layout made the captions easy to overlook during review, we apologize for the inconvenience.

In the revised submission we will **make the placement unambiguous**, for example by: (i) **cross-referencing** each figure in the main text with **“see Figure X caption (Appendix / end of manuscript)”** or the journal’s preferred wording; (ii) ensuring the **submission system’s figure/caption fields** (if used) mirror the same text so captions appear **next to each figure** in the reviewer PDF where the platform allows; and (iii) briefly stating in the **cover letter** that **all figure captions appear on pages […] at the end of the manuscript**. We remain happy to **refine or lengthen** any caption if the reviewer wishes more detail on panels, colour scales, or abbreviations.

---

## Question 2 — Preprocessing: output size and normalization (Line 106)

### *Reviewer comment*
In line 106, “All images were preprocessed to ensure uniform input sizes and consistent normalization for effective model training.” What is the size after preprocessing? What is the method of normalization?

### **Response**
We appreciate this request for specificity and will revise the Methods accordingly.

**Spatial size after preprocessing (exact values used in our study).**  
We will state the following **explicitly** in the revised Methods and, if helpful, in **Table S1**:

| Model / branch | Preprocessed shape (as stored for training) | PyTorch-style tensor passed to **Conv2D** |
|----------------|-----------------------------------------------|-------------------------------------------|
| **Severity (SGI) estimation** | **150 × 120 × 40** (height × width × slice stack) | **(batch, 40, 150, 120)** — **40** axial slices as **input channels** |
| **Fracture detection — axial** | **150 × 120 × 40** | **(batch, 40, 150, 120)** |
| **Fracture detection — coronal** | **256 × 256 × 32** | **(batch, 32, 256, 256)** — **32** coronal slices as **channels** |

Thus the **axial** and **coronal** branches of the fracture network **do not share the same in-plane resolution**: axial uses **150 × 120** and coronal uses **256 × 256**, each after **independent** resampling/cropping to the orbital ROI in that orientation. The **severity** network uses the **same axial grid** as the fracture axial branch (**150 × 120** with **40** slices as channels).

**Normalization.**  
We will clarify that: (1) **DICOM** intensity values were converted using **rescale slope and intercept** where applicable; (2) volumes were clipped to a **soft-tissue / craniofacial HU window** appropriate for orbital CT (we will state the exact **Hounsfield unit** window used, e.g. common bone/soft-tissue ranges if that was our choice); (3) after windowing, intensities were scaled to the **[0, 1]** range using **min–max normalization** (linear rescaling: subtract the minimum and divide by the range so all values lie between 0 and 1); and (4) the **same** preprocessing pipeline (HU rescaling, windowing, and min–max scaling to **[0, 1]**) was applied **train/validation/test** to avoid leakage.

---

## Question 3 — Axial “40 slices”: selection method; coronal (Line 108)

### *Reviewer comment*
In line 108, “We extracted the axial dataset from DICOM files and standardized it by selecting 40 slices per case,” What is the selection method? Automatically or manually? And do you select slices in coronal sequence?

### **Response**
Thank you for raising this important methodological point.

**Axial stack (40 slices).**  
Slice selection was **rule-based and fully automatic**, not manual per slice. We localized the orbit along the axial (superior–inferior) axis using **reference landmarks derived from the orbit**, specifically the **maximum (extent) margin** and the **center margin** of the orbit. From those references, we **adjusted the slice span backward and forward** along the stack (i.e., shifted or expanded the contiguous window about the orbital center / margins according to fixed rules) until **exactly 40 contiguous axial slices** were retained for each case, giving a uniform **channel depth** for the 2D multi-channel CNN. We will add a **short algorithmic description** (or pseudocode) and, if helpful, a figure showing **slice index coverage** relative to the orbit.

**Coronal sequence.**  
For the **fracture-detection** model, **coronal** inputs were prepared **analogously** in terms of **automatic, rule-based** slice selection (**32** slices stacked as channels), but the **in-plane grid** after preprocessing differs from the axial stack: coronal volumes are standardized to **256 × 256** in-plane (**32** slices), versus **150 × 120** (**40** slices) for the axial branch. We will report **32 (coronal)** vs **40 (axial)** separately with the **selection rule** for each orientation.

---

## Question 4 — Reason for horizontal flipping (Lines 122–124)

### *Reviewer comment*
In line 122–124, please kindly explain the reason of doing horizontal flipping?

### **Response**
**Horizontal flipping was not used as data augmentation.** It was used so that **all eyes (and their image stacks) share the same anatomical layout if overlaid or “laid on top of one another”**: the **outer (lateral) side** of every orbit is aligned to the **left** of the image and the **inner (medial) side** to the **right**. This **deterministic canonical orientation** removes **confounding variability** from native left-eye vs right-eye laterality and helps the model learn **location-consistent** patterns (e.g. relative to the lateral vs medial orbital wall) rather than mixing mirror-reversed layouts across samples.

For **axial and coronal** stacks, the same **deterministic** left–right rule was applied **across all slices in the stack** (and, where both planes are used, in a **coordinated** way) so that **channel order** and **slice correspondence** are preserved—only the global in-plane **mirroring** for laterality standardization, not random augmentation.

In the revised manuscript we will describe this under **Preprocessing / orientation standardization** (not under augmentation), state explicitly that **OS vs OD** (or equivalent) dictated whether a flip was applied, and list **any separate augmentation** (e.g. rotation) separately if applicable.

---

## Question 5 — Complete preprocessing flow diagram

### *Reviewer comment*
A complete flow diagram of preprocessing could better illustrate your work. Figure 1 only presents part of it.

### **Response**
We agree. We will add **Figure 1 (revised)** or a new **Figure S1: end-to-end preprocessing pipeline**, including: DICOM ingest → **orientation / spacing** handling → **resampling** (if any) → **ROI definition** (orbital coverage) → **slice extraction** → **HU windowing** → **resize/crop to fixed grids** (**150 × 120** for axial stacks with **40** slices; **256 × 256** for coronal stacks with **32** slices; **150 × 120 × 40** for the **severity** network) → **normalization** → **augmentation (train only)** → **tensor layout (*C*, *H*, *W*)** fed to each model. We will use **distinct swimlanes** for **fracture** vs **SGI** paths if any step differs. This will complement the architecture figure and remove ambiguity about **where** train/validation/test splitting occurs (always **after** patient-level grouping if applicable—we will state patient-level vs case-level splitting explicitly; see also our response to Reviewer 1, Q7).

---

## Question 6 — Unilateral vs bilateral; positive vs negative counts

### *Reviewer comment*
How many cases are unilateral and how many cases are bilateral? How many positive and negative samples in the final dataset?

### **Response**
We thank the reviewer for requesting these descriptive statistics. In the **revised manuscript** we will add a **Table (dataset characteristics)** reporting, for the **final analysed cohort**:

| Quantity | Value (to be filled from study database) |
|----------|--------------------------------------------|
| Total patients / CT studies | … |
| Unilateral orbital involvement | … |
| Bilateral orbital involvement | … |
| **Fracture task:** positive / negative (definition: …) | … / … |
| **SGI / severity task:** severe / non-severe (or class counts) | … / … |

We will define **unilateral vs bilateral** using the **same clinical criteria** as in the text (e.g. fracture or injury **visible on one orbit vs both** on CT). **“Samples”** will be clarified: our models operate at **patient/study level** (one label per stacked input); we will **not** double-count left and right as separate positives unless that was our design (if it was slice-level, we will report that transparently). *Please insert your exact counts from the institutional dataset.*

---

## Question 7 — 200–50 vs 75–25 splits

### *Reviewer comment*
The dataset was split into a 200–50 training-validation split and a 75–25 train-test split. Why not make it consistent?

### **Response**
We thank the reviewer: the **75/25** wording was **inconsistent** with our actual protocol and will be **removed everywhere**.

**Unified split (revision):** The cohort is divided **once** into **training** and **testing** in an **80% / 20%** ratio, with **200** cases for **training** and **50** cases for **testing** (**200 + 50 = 250**; i.e. **80% : 20%**). This **same** split definition (**200 train / 50 test**, **80 / 20**) will be reported **consistently** in the **Methods**, **Results**, **tables**, and **figure captions**—no alternate ratios (e.g. 75/25) unless a **distinct supplementary experiment** is explicitly labeled as such.

We will add **one short paragraph + diagram** in Methods stating **patient-level** (or study-level) splitting, **no leakage** between train and test, and the exact **counts and percentages** above.

---

## Question 8 — Grad-CAM not focusing on fracture site

### *Reviewer comment*
The Grad-CAM image did not focus on the fracture site in fracture samples, and could not provide enough explanations for the users.

### **Response**
We thank the reviewer for this important critique. We address it on **technical** and **clinical** levels.

**Technical limitations of Grad-CAM.**  
Grad-CAM reflects **where gradients of the target logit** w.r.t. **early–mid convolutional feature maps** are large. For **thin** or **partial-volume** fracture lines, **low contrast**, and **stacked-slice-as-channel** inputs, saliency may **diffuse** to **orbital soft tissues, muscle cone, or bone edges** that co-vary with the label, which can **misalign** with the radiologist’s focal fracture line. Our implementation targets **feature maps on the axial branch** (and **dual-branch CAMs** for fracture); we will **disclose the exact layer names** and show **failure cases** as well as successes.

**Revisions we will implement or add:**  
1. **Layer selection:** Compare CAMs from **shallower vs deeper** layers and report **which layer** is used in the main paper with **qualitative agreement** scoring (e.g. radiologist marks ROI; we report overlap).  
2. **Additional interpretability:** **Integrated gradients**, **Guided Grad-CAM**, or **occlusion sensitivity** on small 3D patches to **localize** fractures **in-plane and across adjacent slices**.  
3. **3D context:** Brief discussion or supplementary **3D CAM / attention** if feasible, acknowledging compute.  
4. **Clinical framing:** We will state clearly that CAMs are **hypothesis-generating aids**, **not** diagnostic proof, and that **modest specificity** (see Reviewer 2) limits **standalone** clinical use.

---

# Reviewer 2 — Comments and responses

---

## Question 1 — Modest performance; AUC CI; specificity; benchmarks

### *Reviewer comment*
The overall model performance is modest. The SGI model's AUC is 0.75 with a 95% CI of 0.58–0.87; the lower bound approaches chance level. The fracture detection AUC of 0.83 is also substantially below published benchmarks (>0.95). The authors should discuss these results more transparently, including the clinical implications of the SGI model's 48% specificity.

### **Response**
We thank the reviewer for this careful reading. The **modest absolute performance**, the **wide 95% CI** for SGI (**0.58–0.87** with a lower bound approaching chance), the **gap** between our **fracture AUC (~0.83)** and **published values often >0.95**, and the **clinical meaning of ~48% specificity** for SGI are all **legitimate concerns**. We will address each **explicitly** in the revised Discussion (and, where helpful, Methods) rather than treating metrics as standalone headlines.

**What the reviewer asked us to do (and how we will respond).**  
1. **Interpret modest performance honestly** — we will state that performance is **useful only in context** (task difficulty, data, labels, *n*) and that **external validation** is needed before deployment.  
2. **Interpret the SGI AUC and its CI** — we will keep reporting **point estimate + 95% CI** (bootstrap or DeLong as appropriate), explain that **small test sets and rare positives** widen intervals, and **avoid** implying strong discrimination when the **lower bound is near 0.5**.  
3. **Compare fracture AUC to literature without false equivalence** — we will **not** claim parity with studies that used **easier cohorts or different endpoints**; instead we will **contrast protocols** (below).  
4. **Clinical implications of ~48% specificity** — we will spell out that **many predicted “severe” cases may be false positives** at the chosen threshold, what that means for **triage vs definitive diagnosis**, and how **threshold adjustment** changes sensitivity–specificity trade-offs.

**Contextual reasons our numbers may sit below high benchmarks (transparent Discussion).**  
We will articulate that **direct numeric comparison** to papers reporting **very high AUC** is misleading unless the following are aligned. In our study, several factors **increase difficulty** and **depress apparent performance** relative to many published fracture-AI reports:

| Factor | Why it matters |
|--------|----------------|
| **Smaller dataset** | Fewer training cases and **limited test events** inflate **variance** (wide CIs) and cap how well complex models **generalize**; high literature AUCs often rest on **much larger** curated sets. |
| **More heterogeneous imaging / data** | **Multi-scanner**, variable **protocols**, **reconstruction kernels**, **slice thickness**, and **institutional diversity** increase **domain shift** versus single-site, homogenized datasets used in some benchmark studies. |
| **More difficult cases** | Our cohort may include **subtle**, **comminuted**, or **clinically ambiguous** injuries, **complex trauma**, and **challenging anatomy** rather than **clear-cut** positive and negative examples. |
| **Different ground-truth definition** | **Fracture** and **severe globe injury (SGI)** labels follow **our clinical/radiological criteria** and adjudication rules; they may be **stricter, broader, or differently timed** than in papers the reviewer cites, so **label semantics** differ even when the task name sounds similar. |
| **Class imbalance** | **Few positives** (or imbalanced severity classes) skews **naive accuracy**, complicates **training**, and widens **AUC uncertainty**; some high-AUC studies use **balanced sampling** or **larger minority-class counts**. |
| **Lower effective image quality** | **Noise**, **motion**, **metal**, **partial volumes**, and **resolution limits** on real clinical CT reduce **signal-to-noise** for thin fracture lines and globe injury signs versus **research-quality** or **pre-selected** stacks. |

We will tie these points to the reviewer’s **specificity** concern: in **hard, imbalanced, heterogeneous** data, **operating points** chosen for **screening sensitivity** often yield **modest specificity**; we will discuss **who bears the cost of false positives vs false negatives** in **orbital trauma pathways**.

**Fracture AUC vs literature (structured contrast, not excuse).**  
Beyond the table above, we will **summarize in text** how prior **>0.95** studies differ in **dataset size**, **case mix**, **reference standard** (consensus vs single reader), **whether metrics are slice- vs patient-level**, **external validation**, and **task definition** (e.g. **any fracture** vs **subtle blow-out** only). Our goal is **fair context**, not **diminishing** the reviewer’s point that performance must improve for **clinical reliance**.

**SGI specificity (~48%) — clinical implications.**  
We will state plainly that **specificity ~48%** implies that among patients **without** the SGI-positive criterion (**true negatives**), a **substantial fraction** (~**52%**) may still receive a **positive model prediction** at the reported threshold (**false positives** among negatives). That limits **standalone rule-in** for **severe SGI** and supports use only as **ancillary triage** with **expert review**, **repeat imaging**, or **clinical correlation**. We will report **threshold tuning** options (**Youden**, **fixed high sensitivity**, **utility-based**) and **calibration** where feasible.

---

## Question 2 — Baselines for fracture detection

### *Reviewer comment*
Table 1 compares SGIDetectCNN against other architectures for severity estimation, but no analogous comparison is provided for fracture detection. Given the modest 81% accuracy, baselines for this task are equally necessary.

### **Response**
We agree. We will add **Table 2 (or extend Table 1)** with **fracture-detection baselines** trained on **the same preprocessed tensors** and **splits**, reporting **AUC, accuracy, sensitivity, specificity, F1**, and **95% CIs** where feasible. Candidate baselines: **ResNet-18/34**, **DenseNet**, **EfficientNet-B0** (with **tuned** training as in R2 Q4), and a **simple fusion baseline** (e.g. **late fusion** of separate axial-only and coronal-only networks) to isolate the value of the **dual-branch** design.

---

## Question 3 — Volumetric data vs Conv2D; aggregation; evaluation unit

### *Reviewer comment*
The preprocessing yields volumetric data (e.g., 150×240×40), yet the architectures use only Conv2D layers. How are slice-level features aggregated into an instance-level prediction? Are the 40 slices treated as input channels, or is there a post-hoc aggregation strategy? This must be explicitly described, along with clarification of whether metrics are reported at the slice, eye, or patient level.

### **Response**
We thank the reviewer; this is a **core methodological** point we will make **explicit** in Methods and **Figure 2 (revised)**.

**Direct answer — the 40 slices are input channels.**  
We will state this **unambiguously** in the revised manuscript: on the **axial** pathway, **each of the 40 preprocessed axial slices occupies one position in the input channel dimension** of the **first Conv2D** layer. The model is **not** applied slice-by-slice with later fusion; instead, the tensor **(batch, 40, 150, 120)** is interpreted exactly like a multi-channel image—**C = 40**—so that at every **(row, column)** the network sees a **vector of 40 intensities** (one per slice) and **learns kernels that mix across those channels** from the outset. **There is no separate post-hoc “slice aggregation” block**: channel-wise mixing by **2D convolutions**, together with pooling and the fully connected head, is how information from all **40** slices is integrated into **one** instance-level prediction.

**How 2D convolutions consume a volume.**  
In our fracture model, **multiple slices are stacked along the channel dimension** (not along a separate Conv3D depth axis). After preprocessing, the **axial branch** receives **(batch, 40, 150, 120)** — i.e. **40** axial slices as **channels** on a **150 × 120** spatial grid. The **coronal branch** receives **(batch, 32, 256, 256)** — **32** coronal slices as **channels** on a **256 × 256** grid. The **40** and **32** dimensions are therefore **input-channel depths** for **Conv2D**, analogous to RGB having three channels. **Conv2D** learns filters that **mix information across channels** as well as in-plane, yielding **implicit aggregation across slices** within the network (as elaborated above for the **40**-channel axial case).

**Severity (SGI) model.**  
The **severity** network uses the same **axial-style** preprocessing grid: **(batch, 40, 150, 120)** from a **150 × 120 × 40** volume/stack. Any **auxiliary scalar** (e.g. visual acuity / LogMAR) is **concatenated at the feature level** where applicable, as described in the architecture subsection. We will ensure the **published architecture diagram** matches these **channel counts** and tensor shapes.

**Metrics level.**  
We will state unambiguously that metrics are computed at **patient (study) level** with **one label per input tensor** (*not* per-slice classification in the reported tables), unless we additionally report a **secondary slice-level** analysis in the supplement.

---

## Question 4 — Table 1 baselines; EfficientNet AUC ~0.49; tuning; ImageNet

### *Reviewer comment*
The comparison models in Table 1 appear inadequately tuned. EfficientNet's AUC of 0.49 is below chance, which strongly suggests suboptimal training. Were ImageNet-pretrained weights used? Were hyperparameters individually tuned for each model? Without fair baselines, the claimed superiority of SGIDetectCNN is unconvincing.

### **Response**
We agree that **AUC ≈ 0.49** indicates **failed optimization or protocol mismatch** rather than meaningful model comparison. In the revision we will:

1. **Re-train all baselines** with **per-model hyperparameter search** (learning rate, weight decay, batch size, epochs, **class weights** or **focal loss** if imbalance hurts AUC).  
2. State **clearly** whether **ImageNet pretrained** weights were used for 2D encoders; for **non-natural-image** CT stacks-as-channels, we will discuss **random init vs pretrained first conv adaptation** and use **fair input handling** (e.g. **duplicate weights** for 3→*C* channel inflation or **train from scratch** if that is more appropriate).  
3. Add **learning curves** and **calibration** (reliability diagrams) in the supplement.  
4. If EfficientNet remains weak after tuning, we will **diagnose** (e.g. **collapse to majority class**, **wrong loss**, **label inversion**) and **report** the fix.

**Conclusion:** We will **only** claim superiority of **SGIDetectCNN** where it **exceeds fairly tuned** baselines on the **same splits** with **statistical testing** if *n* allows (e.g. **paired bootstrap** on test predictions).

---

## Question 5 — Architectural detail and ablations

### *Reviewer comment*
The proposed architectures lack sufficient detail. What specifically distinguishes these models from the baselines and accounts for their better performance? An ablation study would help isolate the contribution of key design choices (e.g., dual-branch, DepthwiseConv2D).

### **Response**
We will add: (1) a **full layer table** (kernels, strides, channels, pooling) for **AxCorCNN** and **SGIDetectCNN**; (2) **parameter counts** and **FLOPs** if feasible; (3) **ablation experiments**:

| Ablation | Purpose |
|----------|---------|
| Axial-only | Remove coronal branch |
| Coronal-only | Remove axial branch |
| Early fusion (concatenate stacks) | vs dual-branch late fusion |
| Remove depthwise / large-kernel stage | Test geometric prior |
| Remove clinical scalar (if used) | Quantify clinical input contribution |

We will tie **better performance** (if retained after fair baselines) to **inductive biases**: multi-plane context, **parameter-efficient** depthwise mixing, and **target field-of-view** matching orbital anatomy.

---

## Question 6 — More recent architectures (ViT, DINOv2, Swin, ConvNeXt)

### *Reviewer comment*
The comparison is limited to three dated architectures (EfficientNet, GoogLeNet, VGG16). More recent models, e.g. ViT, DINOv2, Swin Transformer, ConvNeXt, should be included to provide a competitive evaluation.

### **Response**
We agree and will add **modern baselines** where **compute and data scale** permit:

- **ConvNeXt-T/S** and **Swin-T** as strong **hierarchical** baselines.  
- **ViT-S/16** (and optionally **DINOv2** features with a **linear probe** or **small MLP head**) with **careful adaptation** of **channel-stacked CT** to patch embedding (e.g. **per-slice embedding + transformer over slice tokens** for a **fair 3D-aware** comparison).

We will **cite** recent **medical imaging** transformer reviews and discuss **data requirements** and **overfitting risk** on modest *n*. If full fine-tuning is unstable, we will report **linear evaluation** **transparently**.

---

## Question 7 — Why not 3D models?

### *Reviewer comment*
Why not use 3D models instead of 2D models?

### **Response**
We will add a dedicated **subsection** comparing **2D multi-channel** vs **3D CNNs**:

**Reasons we prioritized 2D (stacks as channels):**  
1. **Data efficiency:** 3D models have **more parameters** and often need **larger cohorts**; our sample size favours **strong inductive biases**.  
2. **Anisotropy:** CT **through-plane** spacing often differs from **in-plane**; naive 3D convs require **careful resampling** or **anisotropic** kernels.  
3. **Compute and memory:** Full **3D** high-resolution volumes are **heavier** on **GPU memory** than **multi-channel 2D** stacks at the same in-plane size.  
4. **Clinical workflow:** Radiologists often **scroll** axial and coronal series; **dual 2D** branches **mirror** that practice.

**Acknowledgment:**  
We will report **at least one 3D baseline** (e.g. **ResNet3D-18** or our **ShallowC3D-style** encoder) on **the same ROI** and splits, and discuss **whether 3D** improves **AUC** vs **overfitting**.

---

## Question 8 — Figure clarity and compression

### *Reviewer comment*
Figures are not clear. Especially the characters. (not sure if the online system compressed it)

### **Response**
We will **regenerate all figures** at **higher resolution** (e.g. **600–1200 dpi** for line art where allowed), use **larger font sizes** (≥ **8–10 pt** at final print width), **unify font** (sans-serif), and **avoid colour as the only legend** (add **patterns/labels**). We will upload **vector PDF/SVG** where the journal permits and provide **separate high-resolution** files in supplementary material. We will also **check** the **submission PDF** export settings to **minimize compression artefacts**.

---

## Summary of planned manuscript changes (both reviewers)

| Area | Action |
|------|--------|
| Figures | Full captions; higher resolution; optional flow diagram for preprocessing |
| Methods | Preprocessing grids (**150×120×40** severity & fracture axial; **256×256×32** fracture coronal), HU window, normalization, automatic slice rules, 2D-vs-volume explanation, metrics level; **train/test split 200 / 50 (80% / 20%)** stated consistently |
| Results / tables | Fracture baselines; retuned Table 1; optional modern architectures + 3D baseline |
| Discussion | Honest performance limits, CI, specificity, Grad-CAM limits |
| Experiments | Ablations; fair tuning protocol; interpretability supplements |

---

*Document generated for revision planning. Preprocessing sizes for tensors are set as above; insert institution-specific numbers (dataset counts, HU window details, normalization formula, split counts) where still marked.*
