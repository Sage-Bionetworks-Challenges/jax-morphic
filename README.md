# MorPhiC DREAM Challenge Infrastructure

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

## Baseline Model

> [!NOTE]
> For a full list of available GRN baseline models, see: https://gitlab.com/jaxcomp/tutorial_dream_challenge/-/tree/main?ref_type=heads

The baseline model (`baseline_model/run_model.py`) is adapted from @Poirion's
SCENIC notebook ([Process GRNBoost2], [Process SCENIC]). It predicts
gene regulatory networks (GRNs) separately for each cell type using the
following five-step pipeline:

1. **Load data** — reads single-cell RNA counts, cell type labels, and a curated
   list of known transcription factors (TFs)

2. **Preprocess** — normalizes counts across cells, log-transforms the data, and
   keeps only the top 10,000 most variable genes to reduce noise

3. **GRNBoost2** — for each cell type, uses gradient boosting to score how
   strongly each TF's expression predicts the expression of every other gene,
   producing a ranked list of candidate TF → gene links

4. **SCENIC / RcisTarget** — filters candidate links by checking whether the
   TF's binding motif appears near the gene's promoter in the genome, keeping
   only biologically supported regulatory edges

5. **Output** — saves all predicted TF → gene edges (with importance scores and
   cell type labels) to `predictions.csv`

Participants are encouraged to use this as a starting point and adapt or replace
any step with their own method.

**Usage**

The script is designed so you can develop and test your method iteratively.
By default, it reads input files from `/input` and writes `predictions.csv` to
`/output` to emulate the submission system, but you can adjust these (and other
parameters) to match your local setup:

| Option | Default | Description |
|---|---|---|
| `--input-dir` | `/input` | Path to directory containing input files |
| `--output-dir` | `/output` | Path to directory where `predictions.csv` is written |
| `--nthreads` | `8` | Number of parallel workers for GRNBoost2 |
| `--n-feats` | `10000` | Number of top-variance genes to keep |

Install the dependencies and run the script from the repository root:

```bash
pip install -r baseline_model/requirements.txt

python baseline_model/run_model.py \
    --input-dir input \
    --output-dir output
```

Inspect `output/predictions.csv`, make changes to the script, and re-run as
needed.



[MorPhiC DREAM Challenge]: https://www.synapse.org/morphic_dream
[SynapseWorkflowOrchestrator]: https://github.com/Sage-Bionetworks/SynapseWorkflowOrchestrator
[Process GRNBoost2]: https://gitlab.com/jaxcomp/tutorial_dream_challenge/-/blob/main/README.md?ref_type=heads#process-grnboost2
[Process SCENIC]: https://gitlab.com/jaxcomp/tutorial_dream_challenge/-/blob/main/README.md?ref_type=heads#process-scenic