---
title: MuscleMap
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
        <a class="mm-btn mm-btn-primary" href="installation.html">Get started</a>
        <a class="mm-btn mm-btn-ghost" href="usage.html">View examples</a>
      </div>
      <p class="mm-hero-footnote">
        Built for researchers, clinicians, and data scientists working with whole-body MRI and CT.
      </p>
    </div>
    <div class="mm-hero-banner">
      <img src="logo_musclemap_white.png" alt="MuscleMap Logo">
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
          Designed for whole-body imaging with consistent labelling across muscles and bones,
          enabling large-scale, multi-region analyses.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="musclemap_scroll.gif" alt="MuscleMap whole-body GIF">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Automated quantitative muscle metrics</h3>
        <p>
          Extract muscle-specific metrics such as volumes, intensity or fat-related measures across
          multiple MRI contrasts and CT.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="thresholding_scroll.gif" alt="MuscleMap thresholding GIF">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Registration to muscle template and 3D spatial parametric mapping</h3>
        <p>
          Visualize and calculate the spatial distribution of intramuscular fat in a standardized
          muscle template.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="template.png" alt="Template example">
      </div>
    </div>
    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>Integrated in pipelines</h3>
        <p>
          GitHub GUI and Slicer extension so you can easily run MuscleMap and/or inspect
          individual cases for quality control.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="MuscleMap_Slicer3D.png" alt="MuscleMap Slicer 3D view">
      </div>
    </div>
  </div>
</div>

<div class="mm-section">
  <h2>Quick links</h2>
  <div class="mm-grid mm-grid-2">
    <a class="mm-link-card" href="installation.html">
      <h3>Installation</h3>
      <p>Set up the MuscleMap environment with conda and install the toolbox in editable mode.</p>
    </a>
    <a class="mm-link-card" href="usage.html">
      <h3>Usage</h3>
      <p>Learn how to run <code>mm_segment</code>, <code>mm_extract_metrics</code>, and <code>mm_gui</code>.</p>
    </a>
  </div>
</div>


<div class="mm-section mm-section-muted">
  <h2>Typical workflow</h2>
  <ol class="mm-steps">
    <li><strong>Prepare data</strong> – organise your images and (optionally) metadata.</li>
    <li><strong>Run segmentation</strong> – use <code>mm_segment</code> to generate label maps.</li>
    <li><strong>Extract metrics</strong> – compute quantitative measures with <code>mm_extract_metrics</code>.</li>
    <li><strong>Analyse</strong> – integrate metrics with clinical outcomes or other data sources.</li>
  </ol>
</div>
