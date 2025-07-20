# Online DPO Training Workflow - Complete Mermaid Diagrams

This document contains comprehensive mermaid diagrams visualizing the Online DPO training workflow with detailed function calls.

## Main Training Workflow

```mermaid
graph TB
    subgraph "Initialization Phase"
        A[Start Training] --> B[OnlineDPOTrainer.__init__]
        B --> C[Setup Policy Model]
        C --> D[Setup Reference Model]
        D --> E[Setup Reward Model/Judge]
        E --> F[Setup Tokenizers]
        F --> G[Setup Data Loaders]
        G --> H[Configure Generation]
    end
    
    subgraph "Training Loop"
        H --> I[trainer.train]
        I --> J[Get Training Batch]
        J --> K[training_step]
    end
    
    subgraph "Generation Phase"
        K --> L{Use vLLM?}
        L -->|Yes| M[_generate_vllm]
        L -->|No| N[_generate]
        
        M --> O[Load Latest Weights]
        O --> P[LLM.generate]
        P --> Q[Process vLLM Outputs]
        
        N --> R[_process_inputs_for_generation]
        R --> S[unwrap_model_for_generation]
        S --> T[model.generate]
        T --> U[truncate_right]
    end
    
    subgraph "Scoring Phase"
        Q --> V[Decode Completions]
        U --> V
        V --> W{Judge or Reward Model?}
        
        W -->|Judge| X[Apply Chat Template]
        X --> Y[judge.judge]
        Y --> Z[Convert Ranks to Mask]
        
        W -->|Reward Model| AA[Tokenize Prompts+Completions]
        AA --> BB[Concat Sequences]
        BB --> CC[get_reward]
        CC --> DD[Apply EOS Penalty]
        DD --> EE[Compare Scores]
    end
    
    subgraph "Loss Computation"
        Z --> FF[Select Chosen/Rejected]
        EE --> FF
        FF --> GG[_forward - Policy Model]
        GG --> HH[_forward - Reference Model]
        HH --> II[Compute Log Ratios]
        II --> JJ[Compute DPO Loss]
    end
    
    subgraph "Optimization"
        JJ --> KK[accelerator.backward]
        KK --> LL[optimizer.step]
        LL --> MM[scheduler.step]
    end
    
    subgraph "Logging & Control"
        MM --> NN[Collect Metrics]
        NN --> OO[_maybe_log_save_evaluate]
        OO --> PP{More Batches?}
        PP -->|Yes| J
        PP -->|No| QQ{More Epochs?}
        QQ -->|Yes| I
        QQ -->|No| RR[Save Final Model]
        RR --> SS[End Training]
    end
    
    style A fill:#e1f5fe
    style SS fill:#e8f5e8
    style K fill:#fff3e0
    style JJ fill:#fce4ec
    style P fill:#f3e5f5
    style T fill:#f3e5f5
```

## Detailed Function Call Sequence

```mermaid
sequenceDiagram
    participant Main as Training Script
    participant Trainer as OnlineDPOTrainer
    participant Model as Policy Model
    participant RefModel as Reference Model
    participant RwdModel as Reward Model
    participant Judge as Judge/Scorer
    participant Tokenizer as Tokenizer
    participant DataLoader as DataLoader
    
    Main->>Trainer: __init__(model, ref_model, reward_model, judge, args, ...)
    Trainer->>Model: disable_dropout_in_model()
    Trainer->>RefModel: disable_dropout_in_model()
    Trainer->>RwdModel: .eval()
    Trainer->>Trainer: setup generation_config
    
    Main->>Trainer: train()
    
    loop Training Loop
        Trainer->>DataLoader: get_train_dataloader()
        DataLoader-->>Trainer: batch
        
        Trainer->>Trainer: training_step(model, batch)
        
        alt Use vLLM
            Trainer->>Trainer: _generate_vllm(model, inputs)
            Trainer->>Model: state_dict()
            Trainer->>Trainer: llm.generate(prompts, config)
        else Standard Generation
            Trainer->>Trainer: _generate(model, inputs)
            Trainer->>Trainer: _process_inputs_for_generation(inputs)
            Trainer->>Tokenizer: tokenize_row() / processing
            Trainer->>Model: generate(input_ids, attention_mask, config)
        end
        
        Trainer->>Tokenizer: batch_decode(completion_ids)
        
        alt Judge Scoring
            Trainer->>Judge: judge(prompts, completion_pairs)
            Judge-->>Trainer: rankings
        else Reward Model Scoring
            Trainer->>Tokenizer: tokenize prompts + completions
            Trainer->>RwdModel: forward(prompt_completion_ids)
            Trainer->>Trainer: get_reward()
            RwdModel-->>Trainer: scores
        end
        
        Trainer->>Trainer: _forward(model, prompt_ids, completion_ids)
        Model-->>Trainer: policy_logprobs
        
        alt Reference Model Available
            Trainer->>Trainer: _forward(ref_model, prompt_ids, completion_ids)
            RefModel-->>Trainer: ref_logprobs
        else PEFT Case
            Trainer->>Model: disable_adapter()
            Trainer->>Trainer: _forward(model, prompt_ids, completion_ids)
            Model-->>Trainer: ref_logprobs
            Trainer->>Model: enable_adapter()
        end
        
        Trainer->>Trainer: compute_dpo_loss(policy_logprobs, ref_logprobs, mask)
        Trainer->>Trainer: accelerator.backward(loss)
        Trainer->>Trainer: collect_metrics()
        
        opt Logging Step
            Trainer->>Trainer: _maybe_log_save_evaluate()
            Trainer->>Trainer: log(metrics)
        end
        
        opt Save Step
            Trainer->>Trainer: _save_checkpoint(model)
            Trainer->>Model: save_pretrained()
        end
    end
    
    Trainer->>Model: save_pretrained(output_dir)
    Main-->>Main: Training Complete
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw Prompts] --> B[Chat Template]
        B --> C[Tokenization]
        C --> D[Data Collator]
    end
    
    subgraph "Model Components"
        E[Policy Model] --> F[Generation]
        G[Reference Model] --> H[Log Probs]
        I[Reward Model] --> J[Scoring]
        K[Judge] --> L[Ranking]
    end
    
    subgraph "Generation Pipeline"
        D --> F
        F --> M[Completions]
        M --> N[Decoding]
    end
    
    subgraph "Scoring Pipeline"
        N --> O{Scoring Method}
        O -->|Reward Model| I
        O -->|Judge| K
        J --> P[Preference Mask]
        L --> P
    end
    
    subgraph "Loss Computation"
        P --> Q[Chosen/Rejected Split]
        E --> R[Policy Forward]
        G --> S[Reference Forward]
        Q --> T[Log Ratio Computation]
        R --> T
        S --> T
        T --> U[DPO Loss]
    end
    
    subgraph "Optimization"
        U --> V[Backward Pass]
        V --> W[Gradient Update]
        W --> X[Model Update]
    end
    
    subgraph "Monitoring"
        U --> Y[Loss Metrics]
        P --> Z[Accuracy Metrics]
        T --> AA[KL Metrics]
        Y --> BB[Logging]
        Z --> BB
        AA --> BB
    end
    
    style A fill:#e3f2fd
    style X fill:#e8f5e8
    style U fill:#fce4ec
    style BB fill:#fff3e0
```

## Memory and Computation Flow

```mermaid
graph TD
    subgraph "Memory Layout"
        A[Batch Size B] --> B[Generate 2B Completions]
        B --> C[Policy Logprobs: 2B x Seq]
        C --> D[Reference Logprobs: 2B x Seq]
        D --> E[Preference Mask: B]
    end
    
    subgraph "Computation Flow"
        E --> F[Split Chosen/Rejected]
        F --> G[Chosen Logprobs: B x Seq]
        F --> H[Rejected Logprobs: B x Seq]
        G --> I[Sum Logprobs]
        H --> I
        I --> J[Log Ratios: B]
        J --> K[DPO Loss: Scalar]
    end
    
    subgraph "Gradient Flow"
        K --> L[Loss.backward]
        L --> M[Policy Model Gradients]
        M --> N[Optimizer Step]
        N --> O[Parameter Update]
    end
    
    subgraph "Memory Optimization"
        P[Gradient Checkpointing] --> Q[Reduced Activation Memory]
        R[DeepSpeed ZeRO] --> S[Sharded Parameters]
        T[vLLM] --> U[Efficient Generation]
    end
    
    style K fill:#ffcdd2
    style O fill:#c8e6c9
```

## Error Handling and Edge Cases

```mermaid
graph TB
    subgraph "Input Validation"
        A[Check Model/Ref Model] --> B{Same Object?}
        B -->|Yes| C[Raise ValueError]
        B -->|No| D[Continue]
        
        D --> E{Judge AND Reward Model?}
        E -->|Yes| F[Warn and Use Reward Model]
        E -->|No| G[Continue]
        
        G --> H{Neither Judge nor Reward Model?}
        H -->|Yes| I[Raise ValueError]
        H -->|No| J[Continue]
    end
    
    subgraph "Generation Handling"
        J --> K[Generate Completions]
        K --> L{Contains EOS?}
        L -->|No| M[Apply EOS Penalty]
        L -->|Yes| N[Continue]
        M --> N
        
        N --> O{Sequence Too Long?}
        O -->|Yes| P[Truncate Left]
        O -->|No| Q[Continue]
    end
    
    subgraph "Memory Management"
        P --> Q
        Q --> R{Memory Full?}
        R -->|Yes| S[torch.cuda.empty_cache]
        R -->|No| T[Continue Training]
        S --> T
    end
    
    subgraph "Training Stability"
        T --> U{NaN Loss?}
        U -->|Yes| V[Skip Step / Reduce LR]
        U -->|No| W[Continue]
        
        W --> X{Gradient Norm Too High?}
        X -->|Yes| Y[Clip Gradients]
        X -->|No| Z[Normal Update]
        Y --> Z
    end
    
    style C fill:#ffcdd2
    style I fill:#ffcdd2
    style V fill:#fff3e0
```

## Performance Monitoring

```mermaid
graph LR
    subgraph "Training Metrics"
        A[Loss] --> B[Logging System]
        C[KL Divergence] --> B
        D[Entropy] --> B
        E[Reward Margin] --> B
        F[Accuracy] --> B
    end
    
    subgraph "Model Quality"
        G[RLHF Reward] --> H[Quality Assessment]
        I[EOS Token Rate] --> H
        J[Completion Length] --> H
    end
    
    subgraph "Performance Metrics"
        K[Tokens/Second] --> L[Efficiency Monitoring]
        M[Memory Usage] --> L
        N[GPU Utilization] --> L
    end
    
    subgraph "Output Monitoring"
        B --> O[Wandb/TensorBoard]
        H --> O
        L --> O
        O --> P[Real-time Dashboard]
    end
    
    style O fill:#e1f5fe
    style P fill:#e8f5e8
```

These diagrams provide a complete visual representation of the Online DPO training workflow, covering all aspects from initialization through optimization, including error handling and performance monitoring.