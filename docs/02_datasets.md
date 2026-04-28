# Datasets

## fMRI Data

### Acquisition & Preprocessing

- **Subjects:** 8 participants (sub-02 through sub-09)
- **Sessions:** 3 sessions per subject
- **Runs:** 12 runs per session × 3 sessions = 36 runs per subject
- **Task:** Comprehension task (`task-comp`); participants attend to both image and caption stimuli
- **Format:** BIDS-compliant dataset at `comp_fmri_study_2025/bids`
- **Preprocessing:** fMRIPrep (includes motion correction, slice-timing correction, spatial normalisation, FreeSurfer surface reconstruction)
- **Native space:** All analyses are conducted in subject-native **T1w anatomical space** (not MNI), preserving individual voxel geometry for encoding

### Single-Trial Beta Estimates

Single-trial BOLD response amplitudes are estimated using **GLMsingle** (Kay et al., 2022):

- **Beta type:** Type-D (`TYPED_FITHRF_GLMDENOISE_RR`) — denoised single-trial betas using HRF fitting, GLMdenoise, and ridge regression
- **Output shape:** (X, Y, Z, n_trials) volumetric NIfTI, flattened to (n_voxels, n_trials) for analysis
- **Stimulus ordering:** Extracted from DESIGNINFO to index betas correctly across sessions and runs

### Brain Masking

A whole-brain mask is applied in native T1w space (`space-T1w_desc-brain_mask.nii.gz` from fMRIPrep) to retain only in-brain voxels. This ensures consistency with the native-space betas and avoids analysing background voxels.

### Beta Normalisation

Betas are **z-scored per voxel within each run** before any encoding analysis:

1. For each run, compute the mean and standard deviation of each voxel's responses across its trials.
2. Subtract the mean and divide by the standard deviation.

This removes run-level baseline shifts and amplitude differences that are artefacts of scanner drift or session effects, without affecting the relative stimulus-evoked variance.

---

## Stimuli

### Source: Microsoft COCO

- **Dataset:** COCO 2017 (Lin et al., 2014)
- **Unique stimuli:** 252 COCO images used in the experiment
- **Modalities:** Each stimulus appears in two forms:
  - **Image:** The original COCO photograph
  - **Caption (text):** A single human-written caption describing the image content
- **Metadata file:** CSV mapping `cocoid_x` (COCO image ID) to image filename and caption text

### Design Matrix

A `design_matrix_mapping.csv` file maps each GLMsingle **condition index** to:

- The corresponding COCO ID
- The **modality** of the stimulus (text or image)

This mapping is critical for:
1. Assigning each beta to the correct embedding (image vs. caption)
2. Computing **modality-specific noise ceilings**
3. Separating fMRI responses by stimulus modality in encoding models

### Stimulus Repetitions

Each unique stimulus is presented approximately **6 times** across sessions. Repetitions are used to estimate stimulus reliability (noise ceiling) and are handled by grouping trials by stimulus ID during cross-validation to prevent leakage.

---

## Directory Structure (Relevant Paths)

```text
comp_fmri_study_2025/
├── bids/                          # Raw BIDS data
│   └── sub-{id}/ses-{n}/func/    # events.tsv files per run
├── glmsingle/                     # GLMsingle outputs per subject
│   └── sub-{id}/
│       └── TYPED_FITHRF_GLMDENOISE_RR.npy
└── fmriprep/                      # fMRIPrep outputs
    └── sub-{id}/
        ├── space-T1w_desc-brain_mask.nii.gz
        └── anat/  (FreeSurfer surfaces for visualisation)

data/
├── metadata.csv                   # COCO ID → filepath + caption
└── design_matrix_mapping.csv      # Condition index → COCO ID + modality

models/embeddings/{model}/
├── vision_embeddings/layer_{n}.npy
└── text_embeddings/layer_{n}.npy
```
