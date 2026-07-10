# GPU-Supported Bioinformatics Tools vs Parabricks

---

## Short Answer
```
NO - Parabricks covers only a SPECIFIC set of tools
Many GPU-supported bioinformatics tools
exist OUTSIDE of Parabricks
```

---

## What Parabricks Covers

```
Parabricks focuses on
──────────────────────────────
Germline variant calling pipeline
Somatic variant calling pipeline
Short read sequencing (mainly)

Essentially GPU version of
──────────────────────────────
BWA-MEM2      →  Alignment
GATK4         →  Variant calling
DeepVariant   →  ML variant calling
Mutect2       →  Somatic calling
```

---

## Big Picture - GPU Tools by Category

### Sequencing & Alignment
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
BWA-MEM2          ✅ Yes            Via Parabricks
Minimap2          ✅ Yes            Via Parabricks
STAR              ❌ No             Limited GPU mode
HISAT2            ❌ No             No GPU
Bowtie2           ❌ No             No GPU
```

### Variant Calling
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
HaplotypeCaller   ✅ Yes            Via Parabricks
DeepVariant       ✅ Yes            Also standalone
Mutect2           ✅ Yes            Via Parabricks
FreeBayes         ❌ No             No GPU
Strelka2          ❌ No             No GPU
GATK4             ✅ Yes            Via Parabricks
```

### Nanopore / Long Read
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
Guppy             ❌ No             ✅ GPU native
Dorado            ❌ No             ✅ GPU native
Bonito            ❌ No             ✅ GPU native
Medaka            ❌ No             ✅ GPU native
```

### Genome Assembly
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
MEGAHIT           ❌ No             ✅ GPU native
Flye              ❌ No             Limited
Hifiasm           ❌ No             No GPU
wtdbg2            ❌ No             No GPU
```

### Protein Structure
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
AlphaFold2        ❌ No             ✅ GPU native
RoseTTAFold       ❌ No             ✅ GPU native
ESMFold           ❌ No             ✅ GPU native
GROMACS           ❌ No             ✅ GPU native
AMBER             ❌ No             ✅ GPU native
```

### Machine Learning Based
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
DeepVariant       ✅ Yes            ✅ Also standalone
Clair3            ❌ No             ✅ GPU native
PEPPER-Margin     ❌ No             ✅ GPU native
NanoNet           ❌ No             ✅ GPU native
Basenji           ❌ No             ✅ GPU native
DeepSEA           ❌ No             ✅ GPU native
```

### RNA-seq / Transcriptomics
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
Salmon            ❌ No             No GPU
Kallisto          ❌ No             No GPU
STAR              ❌ No             Limited
StringTie         ❌ No             No GPU
DESeq2            ❌ No             No GPU
```

### Metagenomics
```
Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
Kraken2           ❌ No             Limited GPU
MetaSPAdes        ❌ No             No GPU
DIAMOND           ❌ No             No GPU
MMseqs2           ❌ No             ✅ GPU support
```

---

## Parabricks Coverage Visual

```
All GPU Bioinformatics Tools
┌─────────────────────────────────────────┐
│                                         │
│   ┌─────────────────┐                  │
│   │   Parabricks    │                  │
│   │                 │                  │
│   │  BWA-MEM2       │                  │
│   │  HaplotypeCaller│   Dorado         │
│   │  DeepVariant    │   AlphaFold2     │
│   │  Mutect2        │   MEGAHIT        │
│   │  Minimap2       │   GROMACS        │
│   │                 │   Clair3         │
│   │                 │   MMseqs2        │
│   └─────────────────┘   AMBER         │
│                          etc...        │
└─────────────────────────────────────────┘

Parabricks = small subset of all GPU tools
```

---

## It Depends on Your Pipeline

```
WGS/WES Germline Pipeline
──────────────────────────
QC          →  fastp (CPU)
Align       →  BWA-MEM2 ✅ Parabricks
Variant     →  HaplotypeCaller ✅ Parabricks
Annotate    →  VEP (CPU)
Parabricks covers MOST heavy steps ✅

Nanopore Pipeline
──────────────────────────
Basecall    →  Dorado ❌ NOT in Parabricks
Align       →  Minimap2 ✅ Parabricks
Variant     →  Medaka/Clair3 ❌ NOT in Parabricks
Parabricks covers SOME steps ⚠️

Protein Structure Pipeline
──────────────────────────
MSA         →  HHblits (CPU)
Structure   →  AlphaFold2 ❌ NOT in Parabricks
Dynamics    →  GROMACS ❌ NOT in Parabricks
Parabricks covers NOTHING ❌

Metagenomics Pipeline
──────────────────────────
QC          →  fastp (CPU)
Classify    →  Kraken2 (limited GPU)
Assembly    →  MEGAHIT ❌ NOT in Parabricks
Search      →  MMseqs2 ❌ NOT in Parabricks
Parabricks covers NOTHING ❌
```

---

## Summary

```
Parabricks
──────────────────────────────────────────
✅ Best for WGS/WES germline/somatic
✅ Covers main short read NGS pipeline
✅ Easy to use, optimized by NVIDIA
❌ Does NOT cover Nanopore tools
❌ Does NOT cover protein structure
❌ Does NOT cover metagenomics
❌ Does NOT cover RNA-seq fully
❌ Small subset of all GPU bio tools

Outside Parabricks
──────────────────────────────────────────
Many powerful GPU tools exist
Need to install and manage separately
More flexibility
Covers wider range of analyses
```

---

## Practical Recommendation

```
What are you running?
        │
        ├── WGS/WES pipeline
        │         └──► Parabricks covers you well
        │
        ├── Nanopore sequencing
        │         └──► Use Dorado + Medaka/Clair3
        │              (outside Parabricks)
        │
        ├── Protein structure
        │         └──► AlphaFold2/GROMACS
        │              (outside Parabricks)
        │
        └── Mixed pipeline
                  └──► Combine Parabricks +
                       other GPU tools
```

---

# Clarification: Parabricks RNA-seq Support

---

## You Are Correct to Question This

```
I made an inconsistency
────────────────────────────────
Earlier I said   →  Parabricks supports RNA-seq
Just now I said  →  RNA-seq not covered by Parabricks

Let me clarify properly
```

---

## What Parabricks Actually Supports for RNA-seq

```
Parabricks DOES have RNA-seq related tools
BUT very limited compared to full RNA-seq pipeline
```

### What Parabricks Has
```
pbrun rna_fq2bam    →  RNA-seq alignment (STAR based)
pbrun starfusion    →  Gene fusion detection
```

### What Parabricks Does NOT Have
```
❌ Transcript quantification  (Salmon, Kallisto)
❌ Differential expression    (DESeq2, edgeR)
❌ Transcript assembly        (StringTie)
❌ QC                         (FastQC, fastp)
```

---

## Full RNA-seq Pipeline Reality

```
Step                    Tool            Parabricks?
────────────────────────────────────────────────────
QC & Trimming           fastp           ❌ CPU only
Alignment               STAR            ✅ pbrun rna_fq2bam
Fusion detection        STARFusion      ✅ pbrun starfusion
Quantification          Salmon/Kallisto ❌ CPU only
Differential expression DESeq2/edgeR    ❌ CPU only
Visualization           R/Python        ❌ CPU only
```

---

## Correction to Previous Table

```
CORRECTED RNA-seq / Transcriptomics table

Tool              In Parabricks?    GPU Support
────────────────────────────────────────────────
STAR alignment    ✅ Partial         Via Parabricks
STARFusion        ✅ Yes             Via Parabricks
Salmon            ❌ No              No GPU
Kallisto          ❌ No              No GPU
StringTie         ❌ No              No GPU
DESeq2            ❌ No              No GPU
```

---

## Summary

```
My earlier statement was misleading
────────────────────────────────────────────────
Parabricks DOES support RNA-seq
BUT only the alignment step (STAR)
NOT the full RNA-seq pipeline

So both statements were partially true
but I should have been more precise
```

