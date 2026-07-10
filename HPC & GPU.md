# GPU/Parabricks and HPC Relationship

---

## What is HPC First?

```
HPC = High Performance Computing
    = Cluster of many computers working together
    = Used for large scale scientific computing
```

### HPC Cluster Structure
```
                    ┌─────────────────┐
                    │   Login Node    │  ← You connect here
                    │   (Head Node)   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │   Scheduler     │  ← SLURM / PBS
                    │  (Job Manager)  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────┴──────┐   ┌───────┴─────┐   ┌───────┴─────┐
   │  CPU Node   │   │  CPU Node   │   │  GPU Node   │  ← Has NVIDIA GPU
   │ (compute)   │   │ (compute)   │   │ (compute)   │
   └─────────────┘   └─────────────┘   └─────────────┘
```

---

## How They Relate

```
HPC           =  The infrastructure/cluster
GPU           =  Hardware inside HPC nodes
Parabricks    =  Software that runs ON GPU nodes IN HPC
CUDA          =  Framework to program GPUs IN HPC

They are NOT competing concepts
They WORK TOGETHER
```

### Simple Analogy
```
HPC        →  The building (infrastructure)
GPU Node   →  A room with special equipment
NVIDIA GPU →  The special equipment
Parabricks →  Tool you use with that equipment
CUDA       →  Manual/instructions for the equipment
```

---

## Yes - They Combine Very Well

```
Typical HPC + GPU + Parabricks Setup
─────────────────────────────────────

1. You submit job to HPC scheduler
        │
        ▼
2. Scheduler assigns GPU node
        │
        ▼
3. Parabricks runs on GPU node
        │
        ▼
4. Results saved to shared storage
```

---

## HPC Job Schedulers

```
Most Common Schedulers
──────────────────────
SLURM    →  Most common in modern HPC
PBS/Torque → Older, still used
LSF      →  IBM, used in some institutions
SGE      →  Sun Grid Engine
```

---

## Running Parabricks on HPC with SLURM

### Basic Job Script
```bash
#!/bin/bash
#SBATCH --job-name=parabricks_pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00              # Max runtime
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # CPU cores
#SBATCH --mem=64G                    # RAM
#SBATCH --gres=gpu:1                 # ← Request 1 GPU
#SBATCH --partition=gpu              # ← GPU partition

# Load modules (HPC uses module system)
module load cuda/11.8
module load parabricks/4.1.0

# Run pipeline
pbrun germline \
    --ref /shared/reference/hg38.fa \
    --in-fq /data/sample_R1.fastq.gz /data/sample_R2.fastq.gz \
    --out-bam /results/aligned.bam \
    --out-variants /results/variants.vcf \
    --num-gpus 1
```

### Submit Job
```bash
sbatch pipeline.sh

# Monitor job
squeue -u yourusername
scontrol show job <job_id>
```

---

## Running Parabricks on HPC with PBS

```bash
#!/bin/bash
#PBS -N parabricks_pipeline
#PBS -o logs/output.log
#PBS -e logs/error.log
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1    # ← Request GPU
#PBS -q gpu_queue                              # ← GPU queue

# Load modules
module load cuda/11.8
module load parabricks/4.1.0

cd $PBS_O_WORKDIR

# Run pipeline
pbrun germline \
    --ref /shared/reference/hg38.fa \
    --in-fq /data/sample_R1.fastq.gz /data/sample_R2.fastq.gz \
    --out-bam /results/aligned.bam \
    --out-variants /results/variants.vcf
```

---

## Full HPC Bioinformatics Pipeline

### With Nextflow (Most Common in Bioinformatics)
```groovy
// nextflow.config
profiles {
    hpc_slurm {
        process.executor       = 'slurm'
        process.queue          = 'gpu'
        process.clusterOptions = '--gres=gpu:1'
    }
}

// pipeline.nf
process ALIGN_READS {
    label 'gpu'
    
    input:
    tuple val(sample_id), path(reads)
    
    output:
    tuple val(sample_id), path("${sample_id}.bam")
    
    script:
    """
    pbrun fq2bam \
        --ref ${params.reference} \
        --in-fq ${reads[0]} ${reads[1]} \
        --out-bam ${sample_id}.bam \
        --num-gpus 1
    """
}

process VARIANT_CALL {
    label 'gpu'
    
    input:
    tuple val(sample_id), path(bam)
    
    output:
    tuple val(sample_id), path("${sample_id}.vcf")
    
    script:
    """
    pbrun haplotypecaller \
        --ref ${params.reference} \
        --in-bam ${bam} \
        --out-variants ${sample_id}.vcf \
        --num-gpus 1
    """
}

process FILTER_VARIANTS {
    // CPU step - no GPU needed
    
    input:
    tuple val(sample_id), path(vcf)
    
    output:
    tuple val(sample_id), path("${sample_id}_filtered.vcf")
    
    script:
    """
    bcftools filter \
        -s LowQual \
        -e 'QUAL<20' \
        ${vcf} > ${sample_id}_filtered.vcf
    """
}

// Main workflow
workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    ALIGN_READS(reads_ch)
    VARIANT_CALL(ALIGN_READS.out)
    FILTER_VARIANTS(VARIANT_CALL.out)
}
```

### Run Nextflow on HPC
```bash
# Submit entire pipeline to HPC
nextflow run pipeline.nf \
    -profile hpc_slurm \
    --reads '/data/*_{R1,R2}.fastq.gz' \
    --reference /shared/hg38/reference.fa \
    --outdir /results/
```

---

## With Snakemake (Alternative)

```python
# Snakefile
rule align_reads:
    input:
        r1 = "data/{sample}_R1.fastq.gz",
        r2 = "data/{sample}_R2.fastq.gz"
    output:
        bam = "results/{sample}.bam"
    resources:
        nvidia_gpu = 1,         # ← Request GPU
        mem_mb     = 64000
    shell:
        """
        pbrun fq2bam \
            --ref {params.ref} \
            --in-fq {input.r1} {input.r2} \
            --out-bam {output.bam} \
            --num-gpus 1
        """

rule variant_call:
    input:
        bam = "results/{sample}.bam"
    output:
        vcf = "results/{sample}.vcf"
    resources:
        nvidia_gpu = 1
    shell:
        """
        pbrun haplotypecaller \
            --ref {params.ref} \
            --in-bam {input.bam} \
            --out-variants {output.vcf}
        """
```

### Submit Snakemake to SLURM
```bash
snakemake \
    --executor slurm \
    --default-resources slurm_partition=gpu \
    --jobs 10
```

---

## HPC + GPU Resource Considerations

```
Things to think about
──────────────────────
1. GPU AVAILABILITY
   - GPU nodes are limited & expensive
   - High demand = long queue wait times
   - Request only what you need

2. STORAGE
   - Genomics data is LARGE
   - Use shared/scratch storage
   - Clean up after jobs

3. COST (Cloud HPC)
   - GPU hours cost more than CPU hours
   - AWS/Google/Azure charge per GPU hour
   - Optimize pipeline to minimize GPU time

4. PARALLELISM
   - Run multiple samples simultaneously
   - Each sample on separate GPU job
   - Use job arrays for many samples
```

### Job Array Example (Many Samples)
```bash
#!/bin/bash
#SBATCH --array=1-100          # ← Run 100 samples
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Get sample name from array
SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" samples.txt)

pbrun germline \
    --ref /shared/reference/hg38.fa \
    --in-fq /data/${SAMPLE}_R1.fastq.gz \
              /data/${SAMPLE}_R2.fastq.gz \
    --out-bam /results/${SAMPLE}.bam \
    --out-variants /results/${SAMPLE}.vcf
```

---

## Summary

```
HPC  +  GPU  +  Parabricks  =  Powerful Bioinformatics

HPC         →  Provides the cluster infrastructure
GPU Nodes   →  Special nodes inside HPC with GPUs
Parabricks  →  Runs on GPU nodes via job scheduler
SLURM/PBS   →  Manages and schedules your GPU jobs
Nextflow    →  Orchestrates your entire pipeline on HPC
```

---

## Typical Real World Setup

```
Institution HPC Cluster
├── CPU Nodes (100s)      →  General computation
├── GPU Nodes (10-20)     →  Parabricks, ML jobs
├── High Memory Nodes     →  Assembly, large genomes
└── Shared Storage (PB)   →  All your data
        │
        └── You submit jobs via SLURM
            Parabricks runs on GPU nodes
            Results saved to shared storage
```

---


# Mixing GPU and CPU Nodes in HPC Pipeline

---

## Short Answer
```
YES - You can mix GPU and CPU nodes
This is actually the RECOMMENDED approach
Not everything needs to run on GPU nodes
```

---

## Why Mix GPU and CPU?

```
GPU Nodes are
─────────────
✅ Fast for parallelizable tasks
❌ Expensive
❌ Limited availability
❌ Long queue wait times

CPU Nodes are
─────────────
✅ Abundant
✅ Cheaper
✅ Shorter queue times
❌ Slower for parallel tasks

Strategy = Use GPU only when needed
           Use CPU for everything else
```

---

## Which Steps Go Where?

```
STEP                    WHERE        WHY
────────────────────────────────────────────────────
FastQC / Fastp          CPU Node  ←  Simple, not GPU benefit
BWA-MEM2 alignment      GPU Node  ←  Parabricks, big speedup
Samtools sort/index     CPU Node  ←  No GPU benefit
HaplotypeCaller         GPU Node  ←  Parabricks, big speedup
BCFtools filter         CPU Node  ←  Simple, not GPU benefit
Annotation (VEP)        CPU Node  ←  No GPU benefit
```

---

## How to Specify in SLURM

### GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --partition=gpu          # ← GPU partition
#SBATCH --gres=gpu:1             # ← Request GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module load cuda/11.8
module load parabricks/4.1.0

pbrun fq2bam \
    --ref /shared/ref/hg38.fa \
    --in-fq sample_R1.fastq.gz sample_R2.fastq.gz \
    --out-bam aligned.bam \
    --num-gpus 1
```

### CPU Job
```bash
#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=cpu          # ← CPU partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
                                 # ← No GPU request at all

module load samtools/1.17
module load bcftools/1.17

samtools sort aligned.bam -o sorted.bam
samtools index sorted.bam
bcftools filter -s LowQual -e 'QUAL<20' variants.vcf > filtered.vcf
```

---

## Full Mixed Pipeline with SLURM Dependencies

```bash
#!/bin/bash
# master_pipeline.sh
# Submit jobs with dependencies - each waits for previous to finish

# ─────────────────────────────────
# STEP 1: QC - CPU job
# ─────────────────────────────────
JOB1=$(sbatch --parsable \
    --job-name=qc \
    --partition=cpu \
    --cpus-per-task=8 \
    --mem=16G \
    --wrap="
        module load fastp/0.23
        fastp \
            -i raw_R1.fastq.gz \
            -I raw_R2.fastq.gz \
            -o clean_R1.fastq.gz \
            -O clean_R2.fastq.gz \
            -h qc_report.html
    ")
echo "QC job submitted: $JOB1"

# ─────────────────────────────────
# STEP 2: Alignment - GPU job
# Wait for STEP 1 to finish first
# ─────────────────────────────────
JOB2=$(sbatch --parsable \
    --job-name=alignment \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=64G \
    --dependency=afterok:$JOB1 \       # ← Wait for QC
    --wrap="
        module load cuda/11.8
        module load parabricks/4.1.0
        pbrun fq2bam \
            --ref /shared/ref/hg38.fa \
            --in-fq clean_R1.fastq.gz clean_R2.fastq.gz \
            --out-bam aligned.bam \
            --num-gpus 1
    ")
echo "Alignment job submitted: $JOB2"

# ─────────────────────────────────
# STEP 3: Sort & Index - CPU job
# Wait for STEP 2 to finish first
# ─────────────────────────────────
JOB3=$(sbatch --parsable \
    --job-name=sort_index \
    --partition=cpu \
    --cpus-per-task=8 \
    --mem=32G \
    --dependency=afterok:$JOB2 \       # ← Wait for alignment
    --wrap="
        module load samtools/1.17
        samtools sort aligned.bam -o sorted.bam
        samtools index sorted.bam
    ")
echo "Sort/Index job submitted: $JOB3"

# ─────────────────────────────────
# STEP 4: Variant Calling - GPU job
# Wait for STEP 3 to finish first
# ─────────────────────────────────
JOB4=$(sbatch --parsable \
    --job-name=variant_call \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=64G \
    --dependency=afterok:$JOB3 \       # ← Wait for sort/index
    --wrap="
        module load cuda/11.8
        module load parabricks/4.1.0
        pbrun haplotypecaller \
            --ref /shared/ref/hg38.fa \
            --in-bam sorted.bam \
            --out-variants variants.vcf \
            --num-gpus 1
    ")
echo "Variant calling job submitted: $JOB4"

# ─────────────────────────────────
# STEP 5: Filter & Annotate - CPU job
# Wait for STEP 4 to finish first
# ─────────────────────────────────
JOB5=$(sbatch --parsable \
    --job-name=filter_annotate \
    --partition=cpu \
    --cpus-per-task=8 \
    --mem=32G \
    --dependency=afterok:$JOB4 \       # ← Wait for variant calling
    --wrap="
        module load bcftools/1.17
        module load vep/108
        bcftools filter -s LowQual \
            -e 'QUAL<20' \
            variants.vcf > filtered.vcf
        vep -i filtered.vcf \
            -o annotated.vcf \
            --cache
    ")
echo "Filter/Annotate job submitted: $JOB5"

echo "Pipeline submitted successfully"
echo "Monitor with: squeue -u $USER"
```

### Submit the whole pipeline
```bash
bash master_pipeline.sh

# Output:
# QC job submitted: 12345
# Alignment job submitted: 12346
# Sort/Index job submitted: 12347
# Variant calling job submitted: 12348
# Filter/Annotate job submitted: 12349
# Pipeline submitted successfully
```

---

## Better Approach: Nextflow Handles This Automatically

```groovy
// nextflow.config
profiles {
    hpc {
        process.executor = 'slurm'
        
        // Default is CPU
        process.queue    = 'cpu'
        
        // GPU label overrides to GPU partition
        withLabel: 'gpu' {
            queue        = 'gpu'
            clusterOptions = '--gres=gpu:1'
            module       = 'cuda/11.8:parabricks/4.1.0'
        }
        
        // CPU label stays on CPU partition
        withLabel: 'cpu' {
            queue        = 'cpu'
            module       = 'samtools/1.17:bcftools/1.17'
        }
    }
}

// pipeline.nf
process QC {
    label 'cpu'                    // ← Runs on CPU node
    cpus 8
    memory '16 GB'

    input:
    tuple val(sample), path(reads)

    output:
    tuple val(sample), path("clean_*.fastq.gz")

    script:
    """
    fastp \
        -i ${reads[0]} -I ${reads[1]} \
        -o clean_R1.fastq.gz \
        -O clean_R2.fastq.gz
    """
}

process ALIGN {
    label 'gpu'                    // ← Runs on GPU node
    cpus 16
    memory '64 GB'

    input:
    tuple val(sample), path(reads)

    output:
    tuple val(sample), path("${sample}.bam")

    script:
    """
    pbrun fq2bam \
        --ref ${params.ref} \
        --in-fq ${reads[0]} ${reads[1]} \
        --out-bam ${sample}.bam \
        --num-gpus 1
    """
}

process SORT_INDEX {
    label 'cpu'                    // ← Runs on CPU node
    cpus 8
    memory '32 GB'

    input:
    tuple val(sample), path(bam)

    output:
    tuple val(sample), path("${sample}_sorted.bam")

    script:
    """
    samtools sort ${bam} -o ${sample}_sorted.bam
    samtools index ${sample}_sorted.bam
    """
}

process VARIANT_CALL {
    label 'gpu'                    // ← Runs on GPU node
    cpus 16
    memory '64 GB'

    input:
    tuple val(sample), path(bam)

    output:
    tuple val(sample), path("${sample}.vcf")

    script:
    """
    pbrun haplotypecaller \
        --ref ${params.ref} \
        --in-bam ${bam} \
        --out-variants ${sample}.vcf \
        --num-gpus 1
    """
}

process FILTER {
    label 'cpu'                    // ← Runs on CPU node
    cpus 4
    memory '16 GB'

    input:
    tuple val(sample), path(vcf)

    output:
    tuple val(sample), path("${sample}_filtered.vcf")

    script:
    """
    bcftools filter \
        -s LowQual \
        -e 'QUAL<20' \
        ${vcf} > ${sample}_filtered.vcf
    """
}

// Nextflow automatically handles
// dependencies between processes
workflow {
    reads = Channel.fromFilePairs(params.reads)

    QC(reads)
    ALIGN(QC.out)
    SORT_INDEX(ALIGN.out)
    VARIANT_CALL(SORT_INDEX.out)
    FILTER(VARIANT_CALL.out)
}
```

### Run Nextflow Pipeline
```bash
nextflow run pipeline.nf \
    -profile hpc \
    --reads '/data/*_{R1,R2}.fastq.gz' \
    --ref /shared/ref/hg38.fa \
    --outdir /results/
```

---

## Visual Flow of Mixed Pipeline

```
CPU Node                GPU Node               CPU Node
────────────────────────────────────────────────────────
fastp (QC)
      │
      └──────────────► pbrun fq2bam
                       (alignment)
                              │
                              └──────────────► samtools sort
                                               samtools index
                                                      │
                              ◄─────────────────────┘
                       pbrun haplotypecaller
                       (variant calling)
                              │
                              └──────────────► bcftools filter
                                               vep annotate
                                                      │
                                                      ▼
                                                Final VCF
```

---

## Summary

```
✅ YES you can mix GPU and CPU nodes
✅ Each step submits to appropriate partition
✅ CPU nodes handle simple pre/post processing
✅ GPU nodes handle heavy computation (Parabricks)
✅ SLURM dependencies chain jobs together
✅ Nextflow/Snakemake manages this automatically

Key Point:
Only steps using Parabricks need GPU nodes
Everything else runs on cheaper CPU nodes
```

---
