# Chi tiết luồng xử lý RLHF

## 1. LUỒNG INFERENCE VỚI RLHF

### A. Request Processing Flow
```
┌─────────────┐
│ User Input  │
└─────┬───────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Load Context & Profile (như cũ)                │
│ - Redis: Short-term memory                             │
│ - OpenSearch: User profile vector                      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Tool Selection & Execution (như cũ)            │
│ - Tool Selector Agent                                  │
│ - Execute selected tools (Docs/Web/SQL)               │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Generate Multiple Candidate Responses          │
│ Input: Final prompt với context                        │
│ Output: N responses (N=3-5) với diverse sampling      │
│                                                        │
│ Sampling Strategies:                                   │
│ - Temperature=0.7, top_p=0.9                         │
│ - Temperature=1.0, top_p=0.8                         │
│ - Nucleus sampling với top_k=50                       │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Reward Model Scoring                           │
│                                                        │
│ For each candidate response:                           │
│ Input: [context, query, response]                      │
│ Process:                                               │
│   1. Tokenize input triplet                          │
│   2. Pass through reward model (BERT-based)          │
│   3. Get reward score (0-1 range)                    │
│                                                        │
│ Reward Model Architecture:                             │
│ - Base: roberta-base hoặc bert-base                   │
│ - Input: [CLS] context [SEP] query [SEP] response [SEP]│
│ - Output: Single scalar reward score                   │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Response Selection & Ranking                   │
│                                                        │
│ Selection Strategy:                                    │
│ - Method 1: Chọn response có reward score cao nhất    │
│ - Method 2: Weighted sampling theo reward scores      │
│ - Method 3: Ensemble với rule-based filters           │
│                                                        │
│ Quality Checks:                                        │
│ - Minimum reward threshold (e.g., > 0.6)             │
│ - Safety filters (toxicity, hallucination)           │
│ - Length và coherence checks                          │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Response Delivery với Metadata                 │
│                                                        │
│ Response Package:                                      │
│ {                                                      │
│   "response": "Selected response text",               │
│   "reward_score": 0.85,                              │
│   "response_id": "uuid",                              │
│   "candidates_count": 5,                             │
│   "generation_method": "temperature_0.7",            │
│   "feedback_prompt": true                             │
│ }                                                      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 7: Logging & Storage                              │
│                                                        │
│ Store in MongoDB/PostgreSQL:                           │
│ - conversation_logs table:                            │
│   {                                                    │
│     user_id, session_id, timestamp,                   │
│     query, context, selected_response,                │
│     reward_score, all_candidates                      │
│   }                                                    │
│                                                        │
│ Store in Redis:                                        │
│ - Update conversation history                          │
│ - Cache reward scores cho similar queries             │
└─────────────────────────────────────────────────────────┘
```

### B. Feedback Collection Flow
```
┌─────────────────────────────────────────────────────────┐
│ User Interface - Feedback Components                    │
│                                                        │
│ [Response Display]                                     │
│ "Đây là câu trả lời của tôi..."                      │
│                                                        │
│ [Feedback Section]                                     │
│ 👍 👎  |  ⭐⭐⭐⭐⭐  |  💬 "Add comment"              │
│                                                        │
│ [Advanced Feedback] (Optional)                         │
│ - Accuracy: ⭐⭐⭐⭐⭐                                   │  
│ - Helpfulness: ⭐⭐⭐⭐⭐                               │
│ - Clarity: ⭐⭐⭐⭐⭐                                   │
│ - Safety: ⭐⭐⭐⭐⭐                                    │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ Feedback Processing Service                            │
│                                                        │
│ Input Validation:                                      │
│ - Check user authentication                            │
│ - Validate response_id exists                         │
│ - Rate limiting (max 10 feedback/minute)              │
│                                                        │
│ Data Enrichment:                                       │
│ - Add user metadata (age, location, preferences)      │
│ - Add conversation context                            │
│ - Add timestamp và session info                       │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ Store Feedback Data                                     │
│                                                        │
│ feedback_data table:                                   │
│ {                                                      │
│   feedback_id: uuid,                                  │
│   response_id: uuid,                                  │
│   user_id: string,                                    │
│   session_id: string,                                 │
│   rating_overall: int (1-5),                          │
│   rating_dimensions: {                                │
│     accuracy: int, helpfulness: int,                  │
│     clarity: int, safety: int                         │
│   },                                                   │
│   text_feedback: string,                              │
│   reaction_type: "thumbs_up|thumbs_down|neutral",     │
│   context: {query, previous_messages, user_profile},  │
│   timestamp: datetime,                                │
│   is_training_data: boolean                           │
│ }                                                      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ Real-time Analytics Update                             │
│                                                        │
│ Update metrics in Redis:                               │
│ - Average rating per model version                     │
│ - Feedback distribution                               │
│ - Quality trends over time                            │
│                                                        │
│ Trigger Alerts:                                        │
│ - If rating drops below threshold (< 3.5)            │
│ - If negative feedback spike detected                  │
│ - If safety issues reported                           │
└─────────────────────────────────────────────────────────┘
```

## 2. TRAINING PIPELINE FLOW

### A. Data Preparation Flow
```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Feedback Data Aggregation                      │
│                                                        │
│ Scheduled Job (Daily/Weekly):                          │
│ - Query feedback_data table                           │
│ - Filter: is_training_data = false                    │
│ - Minimum feedback threshold (e.g., 1000 samples)     │
│                                                        │
│ Data Quality Checks:                                   │
│ - Remove spam/bot feedback                            │
│ - Filter extreme outliers                             │
│ - Ensure diverse user representation                   │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Create Training Pairs                          │
│                                                        │
│ Preference Pair Generation:                            │
│ Method 1 - Direct comparison:                          │
│   - Same query, different responses                    │
│   - Compare ratings: response_A vs response_B          │
│   - Label: "A > B" if rating_A > rating_B             │
│                                                        │
│ Method 2 - Absolute scoring:                           │
│   - Single response with rating                       │
│   - Convert to binary: good (>3.5) vs bad (<3.5)     │
│                                                        │
│ Method 3 - Multi-aspect scoring:                       │
│   - Weight different dimensions                        │
│   - Combined score = 0.3*accuracy + 0.3*helpful +     │
│                     0.2*clarity + 0.2*safety          │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Data Preprocessing                             │
│                                                        │
│ Text Processing:                                       │
│ - Tokenization với tokenizer của base model           │
│ - Max length truncation (512 tokens)                  │
│ - Add special tokens: [CLS], [SEP]                    │
│                                                        │
│ Data Format:                                           │
│ {                                                      │
│   "input_ids": [101, 2023, 1005, ...],               │
│   "attention_mask": [1, 1, 1, ...],                  │
│   "labels": 0.85,  # reward score                    │
│   "metadata": {                                       │
│     "user_id": "...", "session_id": "...",          │
│     "feedback_dimensions": {...}                      │
│   }                                                    │
│ }                                                      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Train/Validation Split                         │
│                                                        │
│ Split Strategy:                                        │
│ - 80% training, 20% validation                        │
│ - Stratified split by rating distribution             │
│ - Temporal split: recent data for validation          │
│ - User-based split: some users only in validation     │
│                                                        │
│ Data Augmentation (Optional):                          │
│ - Paraphrase queries với T5/BART                     │
│ - Back-translation cho diversity                      │
│ - Noise injection trong embeddings                    │
└─────────────────────────────────────────────────────────┘
```

### B. Reward Model Training Flow
```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Model Architecture Setup                       │
│                                                        │
│ Base Model: RoBERTa-base hoặc BERT-base               │
│                                                        │
│ Architecture:                                          │
│ Input: [CLS] context [SEP] query [SEP] response [SEP]  │
│   ↓                                                    │
│ RoBERTa Encoder (12 layers)                           │
│   ↓                                                    │
│ [CLS] token representation                             │
│   ↓                                                    │
│ Dropout(0.1)                                          │
│   ↓                                                    │
│ Linear Layer (768 → 1)                                │
│   ↓                                                    │
│ Sigmoid Activation                                     │
│   ↓                                                    │
│ Reward Score (0-1 range)                              │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Training Configuration                         │
│                                                        │
│ Hyperparameters:                                       │
│ - Learning rate: 2e-5                                 │
│ - Batch size: 16                                      │
│ - Max epochs: 5                                       │
│ - Weight decay: 0.01                                  │
│ - Warmup steps: 500                                   │
│                                                        │
│ Loss Function:                                         │
│ - MSE Loss cho absolute scoring                       │
│ - Ranking Loss cho pairwise comparison                │
│ - Combined: α*MSE + β*Ranking                         │
│                                                        │
│ Optimization:                                          │
│ - AdamW optimizer                                     │
│ - Linear warmup + cosine annealing                    │
│ - Gradient clipping (max_norm=1.0)                   │
└─────┬───────────────────────────────────────────────────┘
      ▼ 
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Training Loop                                  │
│                                                        │
│ For each epoch:                                        │
│   For each batch:                                      │
│     1. Forward pass                                    │
│        - Get reward predictions                        │
│        - Calculate loss                               │
│                                                        │
│     2. Backward pass                                   │
│        - Compute gradients                            │
│        - Clip gradients                               │
│        - Update weights                               │
│                                                        │
│     3. Logging                                        │
│        - Loss values                                  │
│        - Learning rate                                │
│        - Gradient norms                               │
│                                                        │
│   End of epoch:                                        │
│     4. Validation                                      │
│        - Calculate validation metrics                  │
│        - Early stopping check                         │
│        - Save checkpoint if best                      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Model Validation & Evaluation                  │
│                                                        │
│ Metrics:                                               │
│ - MSE/MAE cho predicted vs actual ratings             │
│ - Pearson correlation                                  │
│ - Ranking accuracy (pairwise preferences)             │
│ - Calibration plots                                   │
│                                                        │
│ Qualitative Analysis:                                  │
│ - Manual review of high/low scored responses          │
│ - Error analysis by category                          │
│ - Bias detection across user groups                   │
│                                                        │
│ A/B Testing:                                           │
│ - Deploy 10% traffic to new reward model             │
│ - Compare user satisfaction metrics                    │
│ - Statistical significance testing                     │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Model Deployment                               │
│                                                        │
│ Deployment Strategy:                                   │
│ - Blue-Green deployment                               │
│ - Gradual rollout (10% → 50% → 100%)                │
│ - Automatic rollback on metric drops                  │
│                                                        │
│ Model Serving:                                         │
│ - TorchServe hoặc TensorFlow Serving                  │
│ - GPU inference với batch processing                  │
│ - Response time SLA: < 100ms                         │
│                                                        │
│ Monitoring:                                            │
│ - Prediction distribution shifts                       │
│ - Latency và throughput metrics                       │
│ - Model accuracy degradation                          │
└─────────────────────────────────────────────────────────┘
```

### C. Policy Optimization Flow (PPO/DPO)
```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Initialize Policy Training                     │
│                                                        │
│ Components:                                            │
│ - Actor Model: Current LLM (to be fine-tuned)        │
│ - Critic Model: Copy of LLM for value estimation     │
│ - Reference Model: Original LLM (frozen)              │
│ - Reward Model: Trained reward predictor              │
│                                                        │
│ Training Data:                                         │
│ - Sample queries from conversation logs               │
│ - Generate responses với current policy               │
│ - Get reward scores từ reward model                   │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: PPO Training Loop                              │
│                                                        │
│ For each training iteration:                           │
│                                                        │
│ 1. Rollout Phase:                                      │
│    - Sample batch of queries                          │
│    - Generate responses với current policy            │
│    - Get reward scores                                │
│    - Calculate advantages                             │
│                                                        │
│ 2. Policy Update Phase:                               │
│    - Compute policy gradient                          │
│    - Clip importance sampling ratio                    │
│    - Add KL penalty vs reference model                │
│    - Update actor network                             │
│                                                        │
│ 3. Value Update Phase:                                │
│    - Update critic network                            │
│    - Minimize value function loss                     │
│                                                        │
│ Loss Function:                                         │
│ L = L_policy + c1*L_value - c2*entropy + c3*L_KL      │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Safety & Quality Monitoring                    │
│                                                        │
│ During Training:                                       │
│ - Monitor KL divergence vs reference                  │
│ - Check for reward hacking                            │
│ - Validate on held-out test set                      │
│ - Human evaluation on sample outputs                   │
│                                                        │
│ Safety Measures:                                       │
│ - KL penalty coefficient scheduling                   │
│ - Early stopping on quality degradation              │
│ - Periodic human-in-the-loop evaluation              │
│                                                        │
│ Quality Metrics:                                       │
│ - Perplexity vs reference model                       │
│ - Human preference win rate                           │
│ - Task-specific performance                           │
│ - Safety classifier scores                            │
└─────┬───────────────────────────────────────────────────┘
      ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Model Validation & Deployment                  │
│                                                        │
│ Final Evaluation:                                      │
│ - Human evaluation study                              │
│ - A/B test với previous model version                 │
│ - Safety red-teaming                                  │
│ - Performance benchmarking                            │
│                                                        │
│ Deployment Process:                                    │
│ - Shadow mode testing                                 │
│ - Gradual traffic ramp-up                            │
│ - Continuous monitoring                               │
│ - Rollback procedure ready                            │
│                                                        │
│ Success Criteria:                                      │
│ - User satisfaction increase > 5%                     │
│ - No safety incidents                                 │
│ - Response quality maintained                         │
│ - Latency impact < 10%                               │
└─────────────────────────────────────────────────────────┘
```

## 3. MONITORING & MAINTENANCE FLOW

### A. Real-time Monitoring
```
┌─────────────────────────────────────────────────────────┐
│ System Health Monitoring                               │
│                                                        │
│ Key Metrics:                                           │
│ - Response generation latency                          │
│ - Reward model inference time                         │
│ - User feedback rates                                 │
│ - Model prediction accuracy                           │
│                                                        │
│ Alerts:                                               │
│ - Latency > 2 seconds                                │
│ - Feedback rate drop > 20%                           │
│ - Reward score distribution shift                     │
│ - Error rate > 1%                                    │
│                                                        │
│ Dashboards:                                           │
│ - Real-time metrics                                   │
│ - User satisfaction trends                            │
│ - Model performance over time                         │
│ - A/B test results                                   │
└─────────────────────────────────────────────────────────┘
```

### B. Continuous Improvement Loop
```
┌─────────────────────────────────────────────────────────┐
│ Weekly/Monthly Analysis                                │
│                                                        │
│ Data Analysis:                                         │
│ - Feedback pattern analysis                           │
│ - User behavior changes                               │
│ - Model drift detection                               │
│ - Performance regression analysis                      │
│                                                        │
│ Action Items:                                          │
│ - Retrain reward model với new data                   │
│ - Fine-tune policy với recent feedback                │
│ - Update training data filters                        │
│ - Adjust hyperparameters                              │
│                                                        │
│ Experimentation:                                       │
│ - Test new model architectures                        │
│ - Try different training strategies                   │
│ - Evaluate alternative reward functions               │
│ - A/B test UI changes                                │
└─────────────────────────────────────────────────────────┘
```