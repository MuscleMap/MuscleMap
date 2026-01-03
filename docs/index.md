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
        <a class="mm-btn mm-btn-ghost" href="https://github.com/MuscleMap/MuscleMap" target="_blank" rel="noopener">
          View code
        </a>
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
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/muscle-segmentation/' | relative_url }}">
            Whole-body muscle segmentation
          </a>
        </h3>
        <p>
          Designed for whole-body and clinical imaging with consistent labelling
          across muscles and bones, enabling large-scale, multi-region analyses.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/musclemap_scroll.gif' | relative_url }}" alt="MuscleMap whole-body GIF">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/muscle-quantification/' | relative_url }}">
            Automated muscle quantification from MRI and CT
          </a>
        </h3>
        <p>
          Extract muscle-specific metrics such as volumes or intramuscular fat
          percentage across multiple MRI contrasts and CT.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/thresholding.png' | relative_url }}" alt="MuscleMap thresholding png">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/muscle-registration/' | relative_url }}">
            Three-dimensional spatial parametric mapping
          </a>
        </h3>
        <p>
          Visualize and calculate the 3D spatial distribution of intramuscular fat
          in a standardized muscle template.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/template.png' | relative_url }}" alt="Template example">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/slicer-extension/' | relative_url }}">
            Integrated in pipelines
          </a>
        </h3>
        <p>
          Includes a graphical user interface and a Slicer extension to make MuscleMap easy
          to run and inspect individual cases for quality control.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/MuscleMap_Slicer3D.png' | relative_url }}" alt="MuscleMap Slicer 3D view">
      </div>
    </div>
  </div>
</div>
