# MorPhiC DREAM Challenge Evaluation

Repository containing the technical infrastructure and code for the [MorPhiC DREAM Challenge]

The challenge infrastructure is powered by the [SynapseWorkflowOrchestrator]
orchestration tool, which continuously monitors the challenge for new submissions,
automatically processing and evaluating them using the steps defined in `workflow.cwl`.

### Folder structure

```
jax-morphic
├── evaluation      // core scoring and validation scripts
├── README.md
├── scripts         // scripts called by the individual CWL scripts
├── steps           // individual CWL scripts (called by the main workflow CWL)
└── workflow.cwl    // CWL workflow for evaluating submissions

```

## Evaluation Overview

_more details coming soon_

## Baseline GRN Models

See: https://gitlab.com/jaxcomp/tutorial_dream_challenge/-/tree/main?ref_type=heads


[MorPhiC DREAM Challenge]: https://www.synapse.org/morphic_dream
[SynapseWorkflowOrchestrator]: https://github.com/Sage-Bionetworks/SynapseWorkflowOrchestrator
