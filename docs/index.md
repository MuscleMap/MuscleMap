---
title: Home
nav_order: 1
parent: Overview
---

<div class="mm-hero">
  <div class="mm-hero-inner">
    <div class="mm-hero-text">
      <p class="mm-eyebrow">Open-source toolbox</p>
      <h1>MuscleMap</h1>
      <p class="mm-subtitle">
        Whole-body muscle segmentation and quantitative analysis for large-scale imaging studies.
      </p>
      <div class="mm-hero-actions">
        <a class="mm-btn mm-btn-primary" href="{{ '/installation/' | relative_url }}">Get started</a>
        <a class="mm-btn mm-btn-ghost" href="https://github.com/MuscleMap/MuscleMap" target="_blank" rel="noopener"> View code</a>
      </div>
      <p class="mm-hero-footnote">
        Built for researchers, clinicians, and data scientists working with MRI and CT.
      </p>
    </div>
    <div class="mm-hero-banner">
      <img src="{{ '/assets/images/logo_musclemap_white.png' | relative_url }}" alt="MuscleMap whole-body GIF">
    </div>
  </div>
</div>
<div class="mm-section">
  <h2>Features</h2>
  <div class="mm-grid">
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Whole-body muscle segmentation</h3>
        <p>
          Designed for whole-body and clinical imaging with consistent labelling
          across muscles and bones, enabling large-scale, multi-region analyses.
        </p>
        <div class="mm-card-actions">
          <a class="mm-btn mm-btn-ghost mm-btn-sm"
             href="{{ '/muscle-segmentation/' | relative_url }}">
            View mm_segment 
          </a>
        </div>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/musclemap_scroll.gif' | relative_url }}" alt="MuscleMap whole-body GIF">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Automated muscle quantification from MRI and CT</h3>
        <p>
          Extract muscle-specific metrics such as volumes or
          or intramuscular fat percentage across multiple MRI contrasts and CT.
        </p>
        <div class="mm-card-actions">
          <a class="mm-btn mm-btn-ghost mm-btn-sm"
             href="{{ '/muscle-quantification/' | relative_url }}">
            View mm_extract_metrics
          </a>
        </div>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/thresholding.png' | relative_url }}" alt="MuscleMap thresholding png">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3> Three-dimensional spatial parametric mapping</h3>
        <p>
          Visualize and calculate the 3D spatial distribution of intramuscular fat
          in a standardized muscle template.
        </p>
        <div class="mm-card-actions">
          <a class="mm-btn mm-btn-ghost mm-btn-sm"
             href="{{ '/spatial-fat-distribution/' | relative_url }}">
            View mm_register_to_template
          </a>
        </div>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/template.png' | relative_url }}" alt="Template example">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Integrated in pipelines</h3>
        <p>
          Includes a graphical user interface and a Slicer extension to make MuscleMap easy 
          to run and inspect individual cases for quality control.
        </p>
        <div class="mm-card-actions" style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
          <a class="mm-btn mm-btn-ghost mm-btn-sm"
             href="{{ '/graphical-user-interface/' | relative_url }}">
            View mm_gui
          </a>
          <a class="mm-btn mm-btn-ghost mm-btn-sm"
             href="https://github.com/Eddowesselink/SlicerMuscleMap" target="_blank">
            View Slicer extension
          </a>
        </div>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/MuscleMap_Slicer3D.png' | relative_url }}" alt="MuscleMap Slicer 3D view">
      </div>
    </div>
  </div> <!-- einde .mm-grid -->
</div>   <!-- einde Features section -->
