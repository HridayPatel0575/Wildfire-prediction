# FWI-Gated LSTM — Architecture Diagrams
## Paste each mermaid block into https://mermaid.live or mermaid.ai

---

## Diagram 1 — Full Model Architecture

```mermaid
flowchart TD
    INPUT["🌡️ Input Weather Sequence
    Shape: batch × 14 days × 35 features
    ────────────────────────────────
    Base (10): temp, humidity, wind_speed,
    rainfall, cvh, lai_hv, lai_lv, cvl, tvh, tvl
    ────────────────────────────────
    FWI Chain (6): FFMC, DMC, DC, ISI, BUI, FWI
    ────────────────────────────────
    Engineered (19): temp_humidity_interaction,
    dryness_index, fire_danger_index, log_fwi, etc."]

    INPUT -->|"Full 35-feature sequence"| LSTM1
    INPUT -->|"FWI column extracted"| FWI_EXTRACT

    %% ═══════════════════════════════════════════
    %% BRANCH 1 — LSTM
    %% ═══════════════════════════════════════════
    subgraph LSTM_BRANCH ["🔁  BRANCH 1 — LSTM  |  Temporal Pattern Learning"]
        direction TB
        LSTM1["🔁 LSTM Layer 1
        Units: 128
        return_sequences = True
        Learns multi-step temporal patterns
        Output: batch × 14 × 128"]

        DROP1["💧 Dropout  30%
        Regularisation"]

        LSTM2["🔁 LSTM Layer 2
        Units: 64
        return_sequences = False
        Compresses 14-day window → 1 vector
        Output: batch × 64"]

        DROP2["💧 Dropout  30%"]

        DENSE["⬛ Dense  32 units  |  ReLU
        Non-linear feature combination
        Output: batch × 32"]

        DROP3["💧 Dropout  20%"]

        LOGIT_LSTM["📊 logit_lstm
        Dense(1, bias=True)
        Unnormalised LSTM fire score
        Output: batch × 1"]

        LSTM1 --> DROP1 --> LSTM2 --> DROP2 --> DENSE --> DROP3 --> LOGIT_LSTM
    end

    %% ═══════════════════════════════════════════
    %% BRANCH 2 — PHYSICS
    %% ═══════════════════════════════════════════
    subgraph PHYSICS_BRANCH ["🔬  BRANCH 2 — Physics Statistics  |  Zero-Parameter FWI Summary"]
        direction TB

        FWI_EXTRACT["🔬 Extract FWI Column
        fwi = input[:, :, fwi_idx]
        Shape: batch × 14  (one value per day)
        No parameters — pure indexing"]

        STAT1["📈 Stat 1 — Mean FWI
        mean(fwi)
        Overall drought level
        in the 14-day window"]

        STAT2["📈 Stat 2 — Max FWI
        max(fwi)
        Worst single fire-danger day
        Peak severity signal"]

        STAT3["📈 Stat 3 — FWI Trend
        mean(days 8-14) − mean(days 1-7)
        Positive = danger building
        Negative = danger easing
        Strongest learned weight: +0.70"]

        STAT4["📈 Stat 4 — Fraction Elevated Days
        fraction(fwi > +0.5σ)
        Days above fire-risk onset threshold
        Captures sustained elevated danger
        Learned weight: +0.69"]

        STAT5["📈 Stat 5 — FWI Volatility
        mean(|fwi_t − fwi_{t−1}|)
        Day-to-day instability
        High volatility can indicate
        incoming weather breaks
        Learned weight: −0.61"]

        STAT6["📈 Stat 6 — Fraction Extreme Days
        fraction(fwi > +2.0σ)
        Catastrophic fire-weather events
        Distinct from Stat 4
        Separates moderate vs extreme
        Learned weight: −0.11"]

        CONCAT["🔗 Concatenate all 6 statistics
        physics_vec: batch × 6
        ZERO parameters — all pre-computed
        from the raw FWI time-series"]

        BN["⚖️ BatchNormalization
        Scales 6 statistics to comparable range
        Prevents any single stat dominating
        the physics logit gradient"]

        LOGIT_PHYS["📊 logit_physics
        Dense(1, bias=False)
        6 learnable weights — one per statistic
        Interpretable: weight sign shows
        whether stat increases fire risk
        Output: batch × 1"]

        FWI_EXTRACT --> STAT1
        FWI_EXTRACT --> STAT2
        FWI_EXTRACT --> STAT3
        FWI_EXTRACT --> STAT4
        FWI_EXTRACT --> STAT5
        FWI_EXTRACT --> STAT6
        STAT1 --> CONCAT
        STAT2 --> CONCAT
        STAT3 --> CONCAT
        STAT4 --> CONCAT
        STAT5 --> CONCAT
        STAT6 --> CONCAT
        CONCAT --> BN --> LOGIT_PHYS
    end

    %% ═══════════════════════════════════════════
    %% GATE + FUSION
    %% ═══════════════════════════════════════════
    GATE["🔩 Trainable Gate Scalar  α
    tf.Variable — initialised at 0.1
    ─────────────────────────────
    WHY THIS EXISTS:
    Without α, physics Dense(1) receives
    gradient per-parameter 100× larger
    than LSTM branch → LSTM collapses
    → recall declines over training epochs
    ─────────────────────────────
    With α = 0.1 at init:
    Epochs 1-10:  LSTM dominates, learns freely
    Epochs 10+:   α grows, physics opens up
    Result: stable recall curve"]

    SCALED["✕  Gated Physics Contribution
    α × logit_physics
    Physics branch contribution
    is softly throttled at start"]

    FUSION["➕ Residual Fusion
    final_logit = logit_lstm + α·logit_physics
    ─────────────────────────────────────
    logit_lstm  → deep temporal patterns
    α·logit_phys → physics severity boost"]

    SIGMOID["σ  Sigmoid Activation
    P(fire) = 1 / (1 + exp(−final_logit))
    Squashes logit → probability [0, 1]"]

    OUTPUT["🔥 Fire Probability
    0.0 = No Fire
    1.0 = Fire
    ───────────────────────
    Threshold at inference:
    t = 0.10 → Recall = 97.6%
    t = 0.80 → Accuracy = 94.5%"]

    LOGIT_LSTM --> FUSION
    LOGIT_PHYS --> SCALED
    GATE --> SCALED
    SCALED --> FUSION
    FUSION --> SIGMOID --> OUTPUT

    %% ═══════════════════════════════════════════
    %% STYLING
    %% ═══════════════════════════════════════════
    style LSTM_BRANCH fill:#1a3a5c,stroke:#457B9D,color:#fff
    style PHYSICS_BRANCH fill:#3a1a1a,stroke:#E63946,color:#fff
    style GATE fill:#2d2d00,stroke:#f4c21b,color:#fff
    style FUSION fill:#1a3a1a,stroke:#2A9D8F,color:#fff
    style OUTPUT fill:#3a1a00,stroke:#E76F51,color:#fff
    style INPUT fill:#111,stroke:#888,color:#ccc
```

---

## Diagram 2 — Data Pipeline (Sequence Builder)

```mermaid
flowchart TD
    CSV["📁 FLAGED_data.csv
    ~2.3M rows
    2015 – 2020
    180 grid cells (lat/lon pairs)
    Indian Subcontinent"]

    FE["⚙️ Feature Engineering
    10 base meteorological features
    6 FWI physics chain components
    19 engineered interaction features
    ─────────────────────────
    Total: 35 features per row"]

    SEQ["🔄 Sliding Window Builder
    Per (latitude, longitude) location:
    Window = 14 consecutive days → features
    Target = day 15 → fire_flag (0 or 1)
    Stride = 1 day
    ─────────────────────────
    Output: batch × 14 × 35"]

    SPLIT["✂️ Temporal Train / Test Split
    NO shuffling — preserves time order
    ─────────────────────────────────
    Train: 2015–2018
    Test:  2019–2020  ← strict hold-out
    Fire rate ~1.72% in both splits"]

    VALCARVE["✂️ Validation Carve-out
    15% of training sequences set aside
    BEFORE SMOTE and BEFORE scaling
    ─────────────────────────────────
    KEY: val set keeps REAL 1.72% fire rate
    so val_recall during training honestly
    predicts test-set recall"]

    SCALER["📐 StandardScaler
    Fitted ONLY on training fold
    (prevents data leakage into val/test)
    ─────────────────────────────────
    Clip to [−10, +10]
    Replace NaN/Inf with 0"]

    SMOTE["🧬 SMOTE
    Applied to TRAINING FOLD ONLY
    Synthesises minority (fire) examples
    until 50/50 class balance
    ─────────────────────────────────
    Val and Test keep natural 1.72% rate"]

    TRAIN_OUT["📦 X_train / y_train
    Balanced 50/50 fire distribution
    Used for model.fit()"]

    VAL_OUT["📦 X_val / y_val
    Real ~1.72% fire rate
    Used for EarlyStopping / Checkpointing
    val_recall here ≈ test recall"]

    TEST_OUT["📦 X_test / y_test
    2019–2020 hold-out
    Real ~1.72% fire rate
    Used ONLY for final evaluation"]

    CSV --> FE --> SEQ --> SPLIT

    SPLIT -->|"2015-2018"| VALCARVE
    SPLIT -->|"2019-2020 — untouched"| TEST_OUT

    VALCARVE -->|"85% of train"| SCALER
    VALCARVE -->|"15% of train — real distribution"| VAL_OUT

    SCALER --> SMOTE
    SMOTE --> TRAIN_OUT

    style CSV fill:#111,stroke:#888,color:#ccc
    style SMOTE fill:#3a1a1a,stroke:#E63946,color:#fff
    style VALCARVE fill:#1a3a5c,stroke:#457B9D,color:#fff
    style TRAIN_OUT fill:#1a3a1a,stroke:#2A9D8F,color:#fff
    style VAL_OUT fill:#1a3a1a,stroke:#2A9D8F,color:#fff
    style TEST_OUT fill:#2d2d00,stroke:#f4c21b,color:#fff
```

---

## Diagram 3 — Training & Loss

```mermaid
flowchart LR
    subgraph NOVEL ["★ FWI-Gated LSTM — Novel Model"]
        direction TB
        NIN["Predictions p̂"]
        NFL["Physics-Recall Focal Loss
        ─────────────────────────────
        L = FL(y, p̂) + λ·MissPenalty
        ─────────────────────────────
        FL = −αt(1−pt)^γ log(pt)
        γ = 2.0   α = 0.90
        ─────────────────────────────
        MissPenalty = mean(relu(0.5 − p̂)·y)
        Activates ONLY on:
          • True fire events (y=1)
          • Where model underconfident (p̂ < 0.5)
        λ = 0.30
        ─────────────────────────────
        Effect: directly forces logit up
        for missed high-FWI fire events"]
        NIN --> NFL
    end

    subgraph BASELINE ["Baseline — Vanilla LSTM"]
        direction TB
        BIN["Predictions p̂"]
        BFL["Standard Focal Loss
        ─────────────────────────────
        FL = −αt(1−pt)^γ log(pt)
        γ = 2.0   α = 0.90
        ─────────────────────────────
        No physics penalty term
        Cannot distinguish high-FWI vs
        low-FWI missed fires"]
        BIN --> BFL
    end

    subgraph CALLBACKS ["EarlyStopping Strategy"]
        direction TB
        ES1["Primary: monitor = val_auc_pr
        patience = 15
        Stops if ranking quality
        stagnates for 15 epochs"]
        ES2["Secondary: monitor = val_recall
        patience = 20  |  min_delta = 0.005
        Prevents stopping before
        recall has fully matured
        (recall oscillates more than AUC-PR)"]
        LRS["ReduceLROnPlateau
        monitor = val_auc_pr
        factor = 0.5
        patience = 8
        min_lr = 1e-6"]
    end

    subgraph RESULTS ["Final Test Results (2019-2020)"]
        direction TB
        R1["★ FWI-Gated LSTM
        Recall-optimal (t=0.797):
        Accuracy  0.945  Recall  0.379
        AUC-ROC   0.854  AUC-PR  0.124"]
        R2["★ FWI-Gated LSTM — Max Recall (t=0.10):
        Recall    0.976  Precision  0.021"]
        R3["Vanilla LSTM (t=0.797):
        Accuracy  0.944  Recall  0.374
        AUC-ROC   0.849  AUC-PR  0.121"]
    end

    NFL -->|"Adam lr=0.001"| CALLBACKS
    BFL -->|"Adam lr=0.001"| CALLBACKS
    CALLBACKS --> RESULTS

    style NOVEL fill:#3a1a1a,stroke:#E63946,color:#fff
    style BASELINE fill:#1a3a5c,stroke:#457B9D,color:#fff
    style RESULTS fill:#1a3a1a,stroke:#2A9D8F,color:#fff
```

---

## Diagram 4 — Physics Weight Interpretation

```mermaid
flowchart LR
    subgraph WEIGHTS ["Learned Physics Weights (logit_physics Dense layer)"]
        direction TB
        W1["FWI Trend  →  +0.7025 🔴
        Building danger over 14 days
        is the strongest fire predictor"]
        W2["Frac Elevated Days  →  +0.6920 🔴
        Sustained days above +0.5σ
        Second strongest predictor"]
        W3["FWI Volatility  →  −0.6103 🔵
        High swings may indicate
        incoming weather breaks
        reducing fire probability"]
        W4["Max FWI  →  −0.3741 🔵
        Single-day spike without
        sustained trend is less predictive
        (Trend + Frac capture this better)"]
        W5["Mean FWI  →  +0.2387 🔴
        Background drought level
        positive but weaker signal"]
        W6["Frac Extreme Days  →  −0.1091 🔵
        Smallest driver
        Extreme days already captured
        by Trend and Frac Elevated"]
    end

    subgraph INSIGHT ["Key Physical Insight for Paper"]
        I1["Sustained FWI buildup
        (Trend + Fraction)
        is MORE predictive than
        single-day extremes (Max FWI)"]
        I2["This is physically consistent:
        Indian wildfires require
        extended dry spells, not
        isolated hot days"]
    end

    W1 --> I1
    W2 --> I1
    W4 --> I1
    I1 --> I2

    style WEIGHTS fill:#1a1a3a,stroke:#457B9D,color:#fff
    style INSIGHT fill:#3a1a00,stroke:#E76F51,color:#fff
```
