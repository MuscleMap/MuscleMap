---
title: MuscleMap
nav_order: 1
---

<div class="mm-hero">
  <div class="mm-hero-inner">
    <div class="mm-hero-logo">
      <img src="logo.png" alt="MuscleMap Logo">
    </div>

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
  </div>
</div>

<div class="mm-section">
  <h2>Why MuscleMap?</h2>
  <div class="mm-grid">
    <div class="mm-card">
      <h3>Whole-body focus</h3>
      <p>
        Designed for whole-body imaging with consistent labelling across muscles and bones,
        enabling large-scale, multi-region analyses.
      </p>
    </div>
    <div class="mm-card">
      <h3>Quantitative metrics</h3>
      <p>
        Extract region-wise metrics such as volumes and fat-related measures, ready to link with
        clinical, biomechanical, or epidemiological data.
      </p>
    </div>
    <div class="mm-card">
      <h3>Integrates in pipelines</h3>
      <p>
        Command line tools and a GUI so you can both batch-process large cohorts and inspect
        individual cases for quality control.
      </p>
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