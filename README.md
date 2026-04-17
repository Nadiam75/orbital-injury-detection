# orbital-injury-detection
Implementation of CNN-based models with Grad-CAM interpretability for automated detection of orbital fractures and severity classification of ocular injuries using CT imaging.

Fracture Detection 
---------------------------------------
```mermaid

flowchart TD
    A[Axial CT Slice] --> B[Axial Branch<br/>Conv + BN + MaxPool];
    C[Coronal CT Slice] --> D[Coronal Branch<br/>Conv + BN + MaxPool];
    
    B --> E[Axial Feature Map];
    D --> F[Coronal Feature Map];
    
    E --> G[Concatenate Features];
    F --> G;
    
    G --> H[Fully Connected Layer];
    H --> I[Softmax / Output Layer];
    I --> J{Fracture Classification into Fracture / No Fracture};

```

Network Architecture

```mermaid
flowchart TD
    %% Inputs
    AX["Axial input (B, 40, H, W)"]
    COR["Coronal input (B, 32, H, W)"]

    %% Coronal branch
    COR --> C1["Conv2d 32→64 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    C1 --> CP1["MaxPool2d k3 s2 p1"]
    CP1 --> C2["Conv2d 64→32 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    C2 --> CP2["MaxPool2d k3 s2 p1"]
    CP2 --> C3["Conv2d 32→16 k3 s1 p1 → BN → ReLU"]
    C3 --> Cdw1["Depthwise Conv2d 16ch (3×3) s1 p1 → BN → ReLU"]
    Cdw1 --> Cmp["MaxPool2d k3 s2 p1"]
    Cmp --> Cdw2["Depthwise Conv2d 16ch (32×32) s(32,32) → ReLU"]
    Cdw2 --> Cflat["Flatten"]
    Cflat --> Cfc["Linear 16→10 → ReLU"]
    Cfc --> Cfeat["Coronal features (10)"]

    %% Axial branch
    AX --> A1["Conv2d 40→64 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    A1 --> AP1["MaxPool2d k3 s2 p1"]
    AP1 --> A2["Conv2d 64→32 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    A2 --> AP2["MaxPool2d k3 s2 p1"]
    AP2 --> A3["Conv2d 32→16 k3 s1 p1 → BN → ReLU"]
    A3 --> Adw1["Depthwise Conv2d 16ch (3×3) s1 p1 → BN → ReLU"]
    Adw1 --> Amp["MaxPool2d k3 s2 p1"]
    Amp --> Adw2["Depthwise Conv2d 16ch (19×15) s(19,15) → ReLU"]
    Adw2 --> Aflat["Flatten"]
    Aflat --> Afc["Linear 16→10 → ReLU"]
    Afc --> Afeat["Axial features (10)"]

    %% Concatenate + final head
    Afeat --> CAT["Concat: axial(10) + coronal(10) → 20"]
    Cfeat --> CAT
    CAT --> FC["Linear 20→1"]
    FC --> OUT["Output: logit (fracture classification)"]



```





Severity Classification 
---------------------------------------
```mermaid



flowchart TD
    A[Axial CT slice] --> B[Conv → BN → MaxPool × N]
    B --> C[Feature map]
    C --> D[Global average pooling]
    D --> E[Fully connected layer]
    E --> F[Softmax]
    F --> H{Severity Classification into Severe / Non-severe};



```




Network Architecture





```mermaid


flowchart TD
    %% Inputs
    X["Axial input (B, 30, H, W)"]
    V["valog (B, 1)"]

    %% Stem
    X --> S1["Conv2d 30→128 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    S1 --> P1["MaxPool2d k3 s2 p1"]
    P1 --> S2["Conv2d 128→64 k3 s1 p1 → BN → Dropout2d(0.5) → ReLU"]
    S2 --> P2["MaxPool2d k3 s2 p1"]
    P2 --> S3["Conv2d 64→32 k3 s1 p1 → BN → ReLU"]

    %% Depthwise stack
    S3 --> DW1["Depthwise Conv2d 32ch (3×3) s1 p1 → BN → ReLU"]
    DW1 --> P3["MaxPool2d k3 s2 p1"]
    P3 --> DW2["Depthwise Conv2d 32ch (19×15) s(19,15) → ReLU"]

    %% Flatten + concat with valog
    DW2 --> FLAT["Flatten"]
    FLAT --> CAT["Concat with valog (dim=1) → features+1"]
    V --> CAT

    %% Shared FC head
    CAT --> FCAX["Linear 33→10 → ReLU"]

    %% Branching: binary vs multiclass
    FCAX --> M["multiclass?"]
    M --> BHEAD["Binary head: Linear 10→1"]
    M --> MCHEAD["Multiclass head: Linear 10→3"]

    %% Outputs
    BHEAD --> BOUT["Output: severity score (binary)"]
    MCHEAD --> MCOUT["Output: 3-class logits"]



```


