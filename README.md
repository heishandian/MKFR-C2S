# MKFR-C2S Core Code

Core implementation of MKFR-C2S, a multiview knowledge-augmented container scheduling framework.

## Code Structure

```
MKFR-C2S/
├── config.py                     # Global configuration: seeds, torch threads, paths
├── cloudsce/                     # Cloudsce simulation environment
│   ├── specs.py                  #   6 resource dimensions, workload/node specs, hyperparameters
│   ├── node.py                   #   Node model, utilization, balance
│   ├── cluster.py                #   NumPy-vectorized cluster
│   ├── workload.py               #   Workload/Instance, Poisson arrivals, DAG dependencies
│   ├── heatmap.py                #   TS-C2EHM temporal heatmap
│   ├── qos.py                    #   QoS model (RT/TH/FP) and SLA
│   ├── reward.py                 #   Monte-Carlo reward decomposition (credit assignment)
│   ├── env.py                    #   Decision process, reward, observations
│   ├── trace.py                  #   Cluster-trace subset (statistical twin)
│   └── metrics.py                #   Metrics, convergence speed, paired test
├── knowledge/                    # View-level knowledge graphs
│   ├── ctssm.py                  #   Ontology, instruction set, prompt templates
│   ├── kg_builder.py             #   RV/WV/AV/AAV four-view triples, SymbolVocab
│   └── transh.py                 #   TransH knowledge encoder
├── models/
│   ├── mkfr_clnet.py             #   MKFR-CLNet: ViT + cross-modal attention + IIC/IVC losses
│   ├── rainbow.py                #   Rainbow DQN (C51 + dueling pointer head)
│   ├── pg.py                     #   PointerPolicy: A2C / PPO / REINFORCE
│   └── encoders.py               #   State encoders (VBE/IBE/relational/DAG/BiGCN)
├── algorithms/                   # Scheduling algorithms
│   ├── base.py                   #   Base Scheduler / BatchPlanner
│   ├── drl_base.py               #   Shared DRL scheduler skeleton
│   ├── mkfr_c2s.py               #   Two-stage MKFR-C2S
│   ├── smda.py  ddmts.py  mots_aco.py  pso_pga.py        # Heuristics / swarm intelligence
│   ├── coscherrl.py  tvw_rl.py  decima.py  hehifldsa.py  # Representation / semantic learning
│   ├── swarm_utils.py            #   Shared helpers for swarm algorithms
│   └── production.py             #   kube-scheduler / YuniKorn approximators
├── experiments/
│   ├── common.py                 #   Shared rollout / training / evaluation / plotting
│   └── convergence.py            #   Convergence and scalability
└── data/
    ├── prepare_trace.py          #   Generate trace subsets
    ├── raw/                      #   Raw trace files
    └── subset/                   #   Generated trace subsets
```

Results are written to `results/<experiment>/`.
