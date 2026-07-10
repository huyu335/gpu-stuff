# CUDA for Bioinformatics Pipelines

---

## Common Bioinformatics Tasks Suited for GPU

```
✅ Sequence Alignment        - Smith-Waterman, Needleman-Wunsch
✅ Genome Assembly           - De Bruijn graph construction
✅ Short Read Mapping        - BWA, Bowtie equivalent
✅ Variant Calling           - SNP/INDEL detection
✅ RNA-seq Analysis          - Expression quantification
✅ Protein Structure         - Folding, docking
✅ k-mer counting            - Frequency analysis
✅ BLAST-like searches       - Database searching
```

---

## Existing GPU-Accelerated Bioinformatics Tools

### Sequence Alignment
```
NVBIO          - NVIDIA's bioinformatics library (deprecated but educational)
CUDASW++       - Smith-Waterman on GPU
GASAL2         - GPU Accelerated Sequence ALignment library
BWA-MEM2       - Supports GPU acceleration
Parabricks     - NVIDIA's full pipeline (BWA + GATK)
```

### Genome Assembly
```
MEGAHIT        - GPU-accelerated de novo assembler
GPU-Euler      - De Bruijn graph on GPU
```

### Variant Calling
```
NVIDIA Clara Parabricks   - Full GATK4 pipeline on GPU
DeepVariant (GPU)         - Google's deep learning variant caller
```

### Other
```
HMMER (GPU)    - Protein sequence analysis
CUDASW++       - Protein database search
MMseqs2        - Fast sequence search (GPU support)
```

---

## Architecture: GPU Bioinformatics Pipeline

```
Raw Data (FASTQ/FASTA)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  ← Quality trimming, filtering
│   (CPU or GPU)    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Read Alignment   │  ← BWA-MEM2, Parabricks (GPU)
│     (GPU)         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Variant Calling  │  ← GATK4 via Parabricks (GPU)
│     (GPU)         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Post-processing  │  ← Annotation, filtering
│   (CPU or GPU)    │
└───────────────────┘
        │
        ▼
    Results (VCF, BAM, etc.)
```

---

## Option 1: Use NVIDIA Clara Parabricks (Easiest)

### What it does
```
- Full GATK4 pipeline on GPU
- 50x speedup over CPU
- Same results as CPU pipeline
- Supports WGS, WES, RNA-seq
```

### Installation
```bash
# Requires NVIDIA GPU with CUDA 11.0+
# Pull Docker container (recommended)
docker pull nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1

# Or install directly
wget https://developer.nvidia.com/parabricks
```

### Running Parabricks Pipeline
```bash
# Full germline pipeline (FASTQ → VCF)
pbrun germline \
    --ref reference.fa \
    --in-fq sample_R1.fastq.gz sample_R2.fastq.gz \
    --out-bam output.bam \
    --out-variants output.vcf \
    --num-gpus 1

# Just alignment (like BWA-MEM2)
pbrun fq2bam \
    --ref reference.fa \
    --in-fq sample_R1.fastq.gz sample_R2.fastq.gz \
    --out-bam output.bam

# Variant calling on existing BAM
pbrun haplotypecaller \
    --ref reference.fa \
    --in-bam output.bam \
    --out-variants output.vcf
```

---

## Option 2: GASAL2 - Custom Alignment (Intermediate)

### What it does
```
- GPU-accelerated local, global, semi-global alignment
- Integrate into your own C/C++ pipeline
- Very fast for batch alignments
```

### Setup
```bash
git clone https://github.com/nahmedraja/GASAL2.git
cd GASAL2
make GPU_SM_ARCH=sm_86    # Change arch for your GPU
                           # sm_75 = Turing, sm_86 = Ampere
```

### Basic Usage in C++
```cpp
#include "gasal.h"

int main() {
    // Initialize GPU
    gasal_gpu_storage_v gpu_storage_vec;
    gasal_init_streams(&gpu_storage_vec, 
                       max_query_len, 
                       max_target_len, 
                       n_alignments, 
                       GLOBAL,      // Alignment type
                       NULL);

    // Parameters
    Parameters params;
    params.match    =  2;
    params.mismatch = -2;
    params.gap_open = -3;
    params.gap_ext  = -1;

    // Add sequences
    for (int i = 0; i < n_alignments; i++) {
        gasal_pack_strings(query[i],  query_len[i],
                          target[i], target_len[i],
                          &gpu_storage_vec, i);
    }

    // Run alignment on GPU
    gasal_aln_async(&gpu_storage_vec, 
                    n_alignments, 
                    &params, 
                    GLOBAL);

    // Sync and get results
    gasal_synch_streams(&gpu_storage_vec);

    // Extract scores
    for (int i = 0; i < n_alignments; i++) {
        printf("Alignment %d score: %d\n", 
               i, 
               gpu_storage_vec.host_res->score[i]);
    }

    // Cleanup
    gasal_destroy_streams(&gpu_storage_vec);
    return 0;
}
```

---

## Option 3: Write Custom CUDA Kernels

### Smith-Waterman Algorithm on GPU

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

// Scoring parameters
#define MATCH     2
#define MISMATCH -1
#define GAP      -2

// ===== DEVICE HELPER =====
__device__ int max3(int a, int b, int c) {
    return max(a, max(b, c));
}

// ===== SMITH-WATERMAN KERNEL =====
// Each thread handles one query-target pair
__global__ void smithWaterman(
    const char* queries,          // All query sequences
    const char* targets,          // All target sequences  
    const int*  query_lengths,    // Length of each query
    const int*  target_lengths,   // Length of each target
    int*        scores,           // Output scores
    int         max_query_len,
    int         max_target_len,
    int         n_pairs           // Number of pairs
) {
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= n_pairs) return;

    // Get this thread's sequences
    const char* query  = queries  + pair_id * max_query_len;
    const char* target = targets  + pair_id * max_target_len;
    int         qlen   = query_lengths[pair_id];
    int         tlen   = target_lengths[pair_id];

    // Scoring matrix (local, simplified)
    // For real use, allocate in shared or global memory
    int best_score = 0;

    // Simple row-by-row DP
    // Note: for production use anti-diagonal parallelism
    int prev_row[512] = {0};   // Max target length
    int curr_row[512] = {0};

    for (int i = 1; i <= qlen; i++) {
        curr_row[0] = 0;
        for (int j = 1; j <= tlen; j++) {
            int match = prev_row[j-1] + 
                       (query[i-1] == target[j-1] ? MATCH : MISMATCH);
            int del   = prev_row[j]   + GAP;
            int ins   = curr_row[j-1] + GAP;

            curr_row[j] = max3(0, match, max(del, ins));

            if (curr_row[j] > best_score)
                best_score = curr_row[j];
        }
        // Swap rows
        memcpy(prev_row, curr_row, (tlen + 1) * sizeof(int));
    }

    scores[pair_id] = best_score;
}

// ===== HOST CODE =====
int main() {
    // Example sequences
    const char h_queries[]  = "ACGTACGT";
    const char h_targets[]  = "TACGTACG";
    int query_len = 8, target_len = 8;
    int n_pairs   = 1;

    // Device pointers
    char *d_queries, *d_targets;
    int  *d_query_lengths, *d_target_lengths;
    int  *d_scores;

    // Allocate device memory
    cudaMalloc(&d_queries,        n_pairs * query_len  * sizeof(char));
    cudaMalloc(&d_targets,        n_pairs * target_len * sizeof(char));
    cudaMalloc(&d_query_lengths,  n_pairs * sizeof(int));
    cudaMalloc(&d_target_lengths, n_pairs * sizeof(int));
    cudaMalloc(&d_scores,         n_pairs * sizeof(int));

    // Copy to device
    cudaMemcpy(d_queries, h_queries, 
               n_pairs * query_len * sizeof(char), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, 
               n_pairs * target_len * sizeof(char), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_lengths,  &query_len,  sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_lengths, &target_len, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize  = (n_pairs + blockSize - 1) / blockSize;

    smithWaterman<<<gridSize, blockSize>>>(
        d_queries, d_targets,
        d_query_lengths, d_target_lengths,
        d_scores,
        query_len, target_len,
        n_pairs
    );

    cudaDeviceSynchronize();

    // Get results
    int h_score;
    cudaMemcpy(&h_score, d_scores, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Alignment score: %d\n", h_score);

    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_targets);
    cudaFree(d_query_lengths);
    cudaFree(d_target_lengths);
    cudaFree(d_scores);

    return 0;
}
```

### Compile
```bash
nvcc -O3 -arch=sm_86 -o sw_gpu smith_waterman.cu
./sw_gpu
```

---

## Option 4: k-mer Counting on GPU

```cpp
#include <cuda_runtime.h>

#define K 21  // k-mer size

// Encode nucleotide to 2-bit
__device__ uint64_t encodeBase(char c) {
    switch(c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default:  return 0;
    }
}

// Count k-mers - each thread processes one read
__global__ void countKmers(
    const char*     reads,
    const int*      read_lengths,
    unsigned int*   kmer_table,    // Hash table
    int             table_size,
    int             n_reads,
    int             max_read_len
) {
    int read_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_id >= n_reads) return;

    const char* read = reads + read_id * max_read_len;
    int         rlen = read_lengths[read_id];

    // Slide window over read
    for (int i = 0; i <= rlen - K; i++) {
        // Encode k-mer
        uint64_t kmer = 0;
        bool valid = true;
        for (int j = 0; j < K; j++) {
            if (read[i+j] == 'N') { valid = false; break; }
            kmer = (kmer << 2) | encodeBase(read[i+j]);
        }

        if (valid) {
            // Hash and count (atomic to avoid race conditions)
            int hash = kmer % table_size;
            atomicAdd(&kmer_table[hash], 1);
        }
    }
}
```

---

## Full Pipeline Example with CUDA Streams

```cpp
// Use streams to overlap CPU preprocessing with GPU computation

cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Batch 1 on stream1, Batch 2 on stream2 simultaneously
processReads<<<grid, block, 0, stream1>>>(batch1_data, ...);
processReads<<<grid, block, 0, stream2>>>(batch2_data, ...);

// Async memory transfers
cudaMemcpyAsync(d_batch1, h_batch1, bytes, 
                cudaMemcpyHostToDevice, stream1);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

---

## Performance Considerations for Bioinformatics

```
1. SEQUENCE ENCODING
   - Store as 2-bit (A=00, C=01, G=10, T=11)
   - Pack 32 bases per 64-bit integer
   - Reduces memory by 4x

2. MEMORY ACCESS PATTERNS
   - Store sequences in Structure of Arrays (SoA)
   - Not Array of Structures (AoS)
   - Better coalescing

3. BATCH PROCESSING
   - Group sequences of similar length
   - Reduces wasted threads on padding
   - Better GPU utilization

4. ANTI-DIAGONAL PARALLELISM
   - For DP algorithms (SW, NW)
   - Threads work on same anti-diagonal
   - True parallelism in DP matrix

5. LOAD BALANCING
   - Variable length sequences = uneven work
   - Use dynamic scheduling
```

---

## Recommended Setup & Tools

```bash
# System requirements
NVIDIA GPU    >= 16GB VRAM (for genomics)
CUDA Toolkit  >= 11.0
Ubuntu        20.04 / 22.04

# Essential libraries
sudo apt install nvidia-cuda-toolkit
pip install pycuda          # Python CUDA bindings

# Monitoring
nvidia-smi                  # GPU usage
watch -n 1 nvidia-smi       # Live monitoring
nvtop                       # Better live monitor
```

---

## Suggested Learning Path

```
1. ✅ Understand CUDA basics        (done)
2. ⬜ Run Parabricks pipeline        - quick wins
3. ⬜ Explore GASAL2                 - alignment library
4. ⬜ Implement Smith-Waterman       - custom kernel
5. ⬜ Optimize with shared memory    - performance
6. ⬜ Implement k-mer counting       - custom pipeline
7. ⬜ Add CUDA streams               - full async pipeline
```

---

## Useful Resources

```
📄 NVIDIA Parabricks docs    - docs.nvidia.com/clara/parabricks
📄 GASAL2 GitHub             - github.com/nahmedraja/GASAL2
📄 NVBIO (reference)         - nvlabs.github.io/nvbio
📄 BioGPU                    - biogpu.github.io
📄 CUDA Best Practices       - docs.nvidia.com/cuda/cuda-c-best-practices
```


# Answers to Your Questions

---

## Question 1: NVIDIA vs GPU Relationship

### Simple Answer
```
NVIDIA = Company that MAKES the GPU hardware
GPU    = The physical hardware chip
```

### Think of it like this
```
NVIDIA         →    Makes & designs the GPU hardware
GPU (hardware) →    Physical card installed in your machine
CUDA           →    NVIDIA's software/framework to program that GPU
Drivers        →    Software that lets OS talk to GPU hardware
```

### The Correct Setup Order
```
1. Physical GPU card installed in machine
        │
        ▼
2. Install NVIDIA Driver
   (lets your OS recognize the GPU)
        │
        ▼
3. Install CUDA Toolkit
   (lets you write/run GPU programs)
        │
        ▼
4. Install your tools/libraries
   (Parabricks, GASAL2, etc.)
```

### So to answer directly
```
✅ GPU is the hardware  →  you set it up FIRST (physically)
✅ NVIDIA provides      →  drivers + CUDA toolkit (software)
✅ You don't "set up NVIDIA"  
✅ You install NVIDIA drivers to USE your GPU
```

### Check if everything is working
```bash
# Check GPU is detected
lspci | grep -i nvidia

# Check NVIDIA driver installed
nvidia-smi

# Output looks like:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0    |
+-----------------------------------------------------------------------------+
| GPU  Name        | Memory-Usage          | GPU-Util  |
|  0  RTX 3090     | 0MiB / 24268MiB      |    0%     |
+-----------------------------------------------------------------------------+

# Check CUDA installed
nvcc --version
```

### Hardware vs Software Summary
```
HARDWARE (Physical)
└── GPU chip (made by NVIDIA)
    └── Examples: RTX 3090, A100, H100, V100

SOFTWARE (NVIDIA provides)
├── NVIDIA Driver       - OS ↔ GPU communication
├── CUDA Toolkit        - Programming framework
├── cuDNN               - Deep learning library
├── cuBLAS              - Linear algebra library
└── Parabricks          - Bioinformatics tool
```

---

---

## Question 2: Parabricks - Predefined or Custom Pipeline?

### Short Answer
```
Parabricks comes with predefined pipelines
BUT you can combine commands to build your own
```

### What Parabricks Provides (Built-in Pipelines)
```
Germline Pipeline      →  FASTQ → BAM → VCF (SNPs/INDELs)
Somatic Pipeline       →  Tumor vs Normal variant calling
RNA-seq Pipeline       →  Gene expression analysis
DeepVariant Pipeline   →  ML-based variant calling
```

### Predefined Commands Available
```bash
# Alignment
pbrun fq2bam              # FASTQ → BAM (BWA-MEM2 equivalent)
pbrun minimap2            # Long read alignment

# Variant Calling
pbrun haplotypecaller     # GATK HaplotypeCaller
pbrun mutectcaller        # Somatic variant calling
pbrun deepvariant         # DeepVariant

# Full Pipelines
pbrun germline            # Full germline pipeline
pbrun somatic             # Full somatic pipeline

# Utilities
pbrun bammetrics          # BAM statistics
pbrun collectmultiplemetrics
```

### Can You Build Custom Pipeline?
```
YES - by chaining Parabricks commands together
```

```bash
#!/bin/bash
# Example: Your custom pipeline

# Step 1: Alignment
pbrun fq2bam \
    --ref reference.fa \
    --in-fq sample_R1.fastq.gz sample_R2.fastq.gz \
    --out-bam aligned.bam

# Step 2: Your own filtering step (any tool)
samtools view -q 20 -F 4 aligned.bam > filtered.bam
samtools index filtered.bam

# Step 3: Back to Parabricks for variant calling
pbrun haplotypecaller \
    --ref reference.fa \
    --in-bam filtered.bam \
    --out-variants variants.vcf

# Step 4: Your own annotation (any tool)
bcftools annotate variants.vcf > annotated.vcf

# Step 5: Custom filtering
vcftools --vcf annotated.vcf \
         --minQ 30 \
         --out final_variants
```

### Parabricks vs Full Custom Pipeline

| Scenario | Recommendation |
|---|---|
| Standard WGS/WES germline | Use `pbrun germline` |
| Standard somatic | Use `pbrun somatic` |
| Custom steps between | Chain pbrun + other tools |
| Completely custom algorithm | Write CUDA kernels yourself |
| Non-standard organisms | Mix Parabricks + custom tools |

### Mixing Parabricks with Other Tools
```bash
# You can freely mix:
Parabricks commands     →  GPU accelerated steps
Samtools                →  BAM manipulation
BCFtools                →  VCF manipulation
BEDtools                →  Genomic intervals
Custom scripts          →  Your own logic
Snakemake / Nextflow    →  Pipeline management
```

### Practical Example - Custom Metagenomics Pipeline
```bash
#!/bin/bash
# Non-standard pipeline example

# Step 1: QC (not Parabricks)
fastp -i raw_R1.fastq -I raw_R2.fastq \
      -o clean_R1.fastq -O clean_R2.fastq

# Step 2: GPU alignment with Parabricks
pbrun fq2bam \
    --ref host_reference.fa \
    --in-fq clean_R1.fastq clean_R2.fastq \
    --out-bam host_aligned.bam

# Step 3: Extract unmapped reads (your custom step)
samtools view -f 4 host_aligned.bam > unmapped.bam

# Step 4: Assemble unmapped reads (not Parabricks)
megahit -1 unmapped_R1.fastq \
        -2 unmapped_R2.fastq \
        -o assembly/

# Step 5: Your analysis continues...
```

---

## Summary

```
Q1: NVIDIA makes the GPU hardware
    You install their drivers/software to USE the GPU
    Order: Hardware → Driver → CUDA → Tools

Q2: Parabricks has built-in pipelines
    BUT you can mix & match with any other tools
    Use Parabricks for GPU-heavy steps
    Use other tools for everything else
```


# Samtools & External Tools with Parabricks

---

## Short Answer
```
Parabricks ONLY provides GPU-accelerated versions of specific tools
Everything else (Samtools, BCFtools, etc.) you need to install SEPARATELY
```

---

## What Parabricks Includes vs What You Install

```
PARABRICKS (comes with)          YOU INSTALL SEPARATELY
─────────────────────────        ──────────────────────
✅ BWA-MEM2 (as fq2bam)          ❌ Samtools
✅ HaplotypeCaller (GPU)         ❌ BCFtools
✅ DeepVariant (GPU)             ❌ BEDtools
✅ Mutect2 (GPU)                 ❌ FastQC
✅ GATK4 tools (GPU)             ❌ Fastp
✅ Minimap2 (GPU)                ❌ Trimmomatic
                                 ❌ STAR (RNA-seq aligner)
                                 ❌ Any custom scripts
```

---

## Installing Common Bioinformatics Tools

### Option 1: Conda (Recommended - Easiest)
```bash
# Install Miniconda first
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create bioinformatics environment
conda create -n bioenv python=3.9
conda activate bioenv

# Install common tools all at once
conda install -c bioconda \
    samtools \
    bcftools \
    bedtools \
    fastqc \
    fastp \
    trimmomatic \
    star \
    hisat2 \
    vcftools
```

### Option 2: APT (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install \
    samtools \
    bcftools \
    bedtools
```

### Option 3: From Source
```bash
# Samtools example
wget https://github.com/samtools/samtools/releases/download/1.17/samtools-1.17.tar.bz2
tar -xjf samtools-1.17.tar.bz2
cd samtools-1.17
./configure
make
sudo make install
```

---

## How They Work Together

```
Your Pipeline
│
├── Parabricks steps    →  uses GPU  (fast)
│   └── pbrun fq2bam
│   └── pbrun haplotypecaller
│
└── External tool steps →  uses CPU  (normal speed)
    └── samtools sort
    └── samtools index
    └── bcftools filter
```

### Practical Example
```bash
#!/bin/bash

# PARABRICKS - GPU accelerated (comes with Parabricks)
pbrun fq2bam \
    --ref reference.fa \
    --in-fq sample_R1.fastq.gz sample_R2.fastq.gz \
    --out-bam aligned.bam

# SAMTOOLS - CPU (you installed separately)
samtools sort aligned.bam -o sorted.bam
samtools index sorted.bam

# PARABRICKS - GPU accelerated again
pbrun haplotypecaller \
    --ref reference.fa \
    --in-bam sorted.bam \
    --out-variants output.vcf

# BCFTOOLS - CPU (you installed separately)
bcftools filter -s LowQual \
    -e 'QUAL<20' \
    output.vcf > filtered.vcf
```

---

## Check What You Have Installed

```bash
# Check each tool
which samtools    && samtools --version
which bcftools    && bcftools --version
which bedtools    && bedtools --version
which fastqc      && fastqc --version
which pbrun       && pbrun version        # Parabricks
```

---

## Simple Rule to Remember

```
Ask yourself:
"Is this tool listed in pbrun --list?"
        │
        ├── YES  →  Parabricks handles it (GPU)
        │
        └── NO   →  You need to install it separately (CPU)
```

```bash
# See all Parabricks available commands
pbrun --help
```

---

## Recommended Setup for Bioinformatics

```bash
# 1. Install Parabricks (GPU tools)
# 2. Install Conda
# 3. Create environment with CPU tools

conda create -n bioenv python=3.9
conda activate bioenv
conda install -c bioconda samtools bcftools bedtools fastqc fastp

# Now you have everything needed
# GPU steps  → pbrun
# CPU steps  → conda environment tools
```


# Parabricks Tool Versions

---

## Short Answer
```
YES - Parabricks bundles specific versions of tools
You have LIMITED ability to choose versions
Version is tied to Parabricks release
```

---

## How Parabricks Versions Work

```
Parabricks Version  →  Bundles Specific Tool Versions
─────────────────────────────────────────────────────
Parabricks 4.0      →  GATK 4.2.4, BWA-MEM2 2.2.1, etc.
Parabricks 4.1      →  GATK 4.2.6, BWA-MEM2 2.2.1, etc.
Parabricks 4.2      →  GATK 4.3.0, BWA-MEM2 2.2.1, etc.
```

### Check What Versions Are Bundled
```bash
# Check Parabricks version
pbrun version

# Output example:
# Parabricks Version 4.1.0-1
# GATK Version: 4.2.6.1
# BWA-MEM2 Version: 2.2.1
```

---

## Tool Versions by Parabricks Release

| Parabricks | GATK | BWA-MEM2 | DeepVariant | CUDA |
|---|---|---|---|---|
| 4.0 | 4.2.4 | 2.2.1 | 1.4 | 11.0+ |
| 4.1 | 4.2.6 | 2.2.1 | 1.5 | 11.8+ |
| 4.2 | 4.3.0 | 2.2.1 | 1.6 | 12.0+ |

```bash
# Always check official docs for exact versions
# https://docs.nvidia.com/clara/parabricks/latest/releaseNotes.html
```

---

## Can You Choose Versions?

### Option 1: Choose Parabricks Version (Limited Control)
```bash
# Pull specific Parabricks version via Docker
docker pull nvcr.io/nvidia/clara/clara-parabricks:4.0.0-1
docker pull nvcr.io/nvidia/clara/clara-parabricks:4.1.0-1
docker pull nvcr.io/nvidia/clara/clara-parabricks:4.2.0-1

# Run specific version
docker run --gpus all \
    nvcr.io/nvidia/clara/clara-parabricks:4.1.0-1 \
    pbrun version
```

### Option 2: Use CPU Version of Tool (Full Control)
```bash
# If you need specific GATK version
# Install it separately and use CPU version instead

# Install specific GATK version
conda install -c bioconda gatk4=4.2.0.0

# Use CPU GATK instead of Parabricks
gatk HaplotypeCaller \
    -R reference.fa \
    -I input.bam \
    -O output.vcf
# ⚠️ This loses GPU acceleration
```

### Option 3: Mix Versions (Recommended Approach)
```bash
#!/bin/bash

# GPU steps  → use Parabricks (bundled versions)
pbrun fq2bam \
    --ref reference.fa \
    --in-fq R1.fastq.gz R2.fastq.gz \
    --out-bam aligned.bam

# CPU steps  → use YOUR specific version
# e.g. specific GATK version you need
gatk-4.1.9.0 HaplotypeCaller \
    -R reference.fa \
    -I aligned.bam \
    -O output.vcf
```

---

## Checking Bundled Tool Versions

```bash
# Inside Parabricks Docker container
docker run --gpus all --rm \
    nvcr.io/nvidia/clara/clara-parabricks:4.1.0-1 \
    bash -c "pbrun version"

# Check GATK version specifically
docker run --gpus all --rm \
    nvcr.io/nvidia/clara/clara-parabricks:4.1.0-1 \
    bash -c "gatk --version"
```

---

## Version Compatibility Concern?

### Common Scenario
```
Your lab/project requires  →  GATK 4.1.9
Parabricks 4.2 bundles     →  GATK 4.3.0
        │
        ▼
Options:
1. Use older Parabricks version  →  has GATK 4.1.x
2. Use CPU GATK 4.1.9            →  lose GPU speed
3. Check if results are same     →  often they are
```

### Check Result Compatibility
```bash
# Run same data on both versions
# Compare outputs

# Parabricks version
pbrun haplotypecaller \
    --ref ref.fa \
    --in-bam input.bam \
    --out-variants parabricks_output.vcf

# CPU GATK specific version
gatk HaplotypeCaller \
    -R ref.fa \
    -I input.bam \
    -O cpu_output.vcf

# Compare
bcftools stats parabricks_output.vcf > parabricks_stats.txt
bcftools stats cpu_output.vcf        > cpu_stats.txt
diff parabricks_stats.txt cpu_stats.txt
```

---

## Summary

```
✅ Parabricks bundles specific tool versions
✅ Version tied to Parabricks release
✅ You can pin Parabricks version via Docker
✅ You can use CPU tools for specific version needs
⚠️  Cannot change individual tool versions INSIDE Parabricks
⚠️  Changing to CPU loses GPU acceleration
```

---

## Recommendation

```
1. Check if bundled version meets your needs
        │
        ├── YES  →  Use Parabricks as is
        │
        └── NO   →
                 ├── Try older Parabricks version
                 │
                 └── Use CPU tool for that specific step
                     keep Parabricks for other steps
```



# Why Load Both CUDA and Parabricks?

---

## Short Answer
```
Parabricks  =  The bioinformatics application
CUDA        =  The underlying engine Parabricks NEEDS to run

Parabricks cannot run WITHOUT CUDA
They are separate software layers
```

---

### More Technical Analogy
```
CUDA        →  Like Java Runtime Environment (JRE)
Parabricks  →  Like a Java Application

Java app cannot run without JRE installed
Parabricks cannot run without CUDA installed
```

---

## Software Layer Visualization

```
┌─────────────────────────┐
│      Parabricks         │  ← You use this
├─────────────────────────┤
│      CUDA Toolkit       │  ← Parabricks uses this
├─────────────────────────┤
│    NVIDIA Driver        │  ← CUDA uses this
├─────────────────────────┤
│    GPU Hardware         │  ← Driver talks to this
└─────────────────────────┘

Each layer DEPENDS on the layer below it
```

---

## Why Separate in HPC Module System?

### HPC Module System Logic
```
HPC clusters have MANY software versions installed
Module system lets users load SPECIFIC versions
They are kept separate intentionally for flexibility
```

```bash
# HPC might have multiple versions available
module avail cuda
# cuda/10.2
# cuda/11.0
# cuda/11.8    ← you pick this one
# cuda/12.0

module avail parabricks
# parabricks/3.8.0
# parabricks/4.0.0
# parabricks/4.1.0    ← you pick this one
# parabricks/4.2.0
```

### Why Flexibility Matters
```
Different tools need different CUDA versions
─────────────────────────────────────────
Parabricks 4.1  →  needs CUDA 11.8
Parabricks 4.2  →  needs CUDA 12.0
Some ML tool    →  needs CUDA 11.0

If bundled together, you couldn't mix versions
Keeping separate = more flexible
```

---

## Could They Be Combined Into One Module?

```
YES - HPC admins COULD create a combined module
Some HPC clusters do this

# Some clusters let you just do:
module load parabricks/4.1.0
# and it auto-loads correct CUDA version

# Check if your HPC does this
module show parabricks/4.1.0
# Will show you what dependencies it loads
```

---

## Summary

```
You load CUDA        →  Provides GPU programming libraries
You load Parabricks  →  Provides bioinformatics tools

Parabricks sits ON TOP of CUDA
Both needed because they are separate software layers
Kept separate in HPC for version flexibility

Think of it as:
CUDA        =  Foundation
Parabricks  =  Built on that foundation
```



