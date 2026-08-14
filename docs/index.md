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
        <a class="mm-btn mm-btn-red" href="https://musclemap-3d.eddo-wesselink.workers.dev/" target="_blank" rel="noopener">
          View MuscleMap in 3D
        </a>
      </div>
      <p class="mm-hero-footnote">
        Built for researchers, clinicians, and data scientists working with MRI and CT.
      </p>
    </div>
    <div class="mm-hero-banner">

      <!-- ===== 3D-vooraanzicht met CT/MRI-schakelaar =====================
           Om het oude logo terug te zetten: zet dit blok tussen commentaar en
           haal het commentaar weg van het blok "OUD LOGO" hieronder. -->
      <div class="mm-modelswitch">
        <img class="mm-model is-on" data-model="ct"
             src="{{ '/assets/images/hero_ct_front.png' | relative_url }}"
             alt="Whole-body CT muscle segmentation, front view">
        <img class="mm-model" data-model="mri"
             src="{{ '/assets/images/hero_mri_front.png' | relative_url }}"
             alt="Whole-body MRI muscle segmentation, front view">
        <div class="mm-modelbtns" role="group" aria-label="Imaging modality">
          <button type="button" class="is-on" data-model="ct" aria-pressed="true">CT</button>
          <button type="button" data-model="mri" aria-pressed="false">MRI</button>
        </div>
      </div>
      <!-- ===== einde 3D-vooraanzicht ================================== -->

      <!-- ===== OUD LOGO ==============================================
      <img src="{{ '/assets/images/logo_musclemap_white.png' | relative_url }}" alt="MuscleMap whole-body GIF">
           ===== einde OUD LOGO ======================================== -->

    </div>
  </div>
</div>

<div class="mm-hero mm-hero-axial">
  <div class="mm-hero-inner">
    <div class="mm-hero-text">
      <h2>Whole-body muscle segmentation</h2>
      <p class="mm-subtitle">
        Designed for whole-body and clinical imaging with consistent labelling
        across muscles and bones, enabling large-scale, multi-region analyses.
      </p>
    </div>
    <div class="mm-hero-banner">
      <div class="mm-modelswitch">
        <img class="mm-model is-on" data-model="ct"
             src="{{ '/assets/images/musclemap_scroll_ct.gif' | relative_url }}"
             alt="Scrolling axial CT with MuscleMap muscle segmentation overlay">
        <img class="mm-model" data-model="mri"
             src="{{ '/assets/images/musclemap_scroll.gif' | relative_url }}"
             alt="Scrolling axial MRI with MuscleMap muscle segmentation overlay">
        <div class="mm-modelbtns" role="group" aria-label="Imaging modality">
          <button type="button" class="is-on" data-model="ct" aria-pressed="true">CT</button>
          <button type="button" data-model="mri" aria-pressed="false">MRI</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Bedient alle CT/MRI-schakelaars op deze pagina; staat bewust buiten de
     blokken hierboven, zodat het blijft werken als je die uitcommentarieert. -->
<script src="{{ '/assets/js/model-switch.js' | relative_url }}" defer></script>

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
        <!-- Geen eigen knoppen: volgt de modaliteit die elders op de pagina is gekozen. -->
        <div class="mm-modelswitch">
          <img class="mm-model is-on" data-model="ct"
               src="{{ '/assets/images/musclemap_scroll_ct.gif' | relative_url }}"
               alt="Scrolling axial CT with MuscleMap muscle segmentation overlay">
          <img class="mm-model" data-model="mri"
               src="{{ '/assets/images/musclemap_scroll.gif' | relative_url }}"
               alt="Scrolling axial MRI with MuscleMap muscle segmentation overlay">
        </div>
      </div>
    </div>

    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/muscle-quantification/' | relative_url }}">
            Automated muscle quantification
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
            3D Slicer extension
          </a>
        </h3>
        <p>
          Includes a 3D Slicer extension to make MuscleMap easy
          to run and inspect individual cases for quality control.
        </p>
      </div>
      <div class="mm-card-media">
        <img src="{{ '/assets/images/MuscleMap_Slicer3D.png' | relative_url }}" alt="MuscleMap Slicer 3D view">
      </div>
    </div>

    <div class="mm-card mm-card-with-media">
      <div class="mm-card-text">
        <h3>
          <a class="mm-btn mm-btn-ghost"
             href="{{ '/neurodesk/' | relative_url }}">
            Neurodesk web application
          </a>
        </h3>
        <p>
          Run MuscleMap directly in your browser through Neurodesk.
          Simply drag and drop NIfTI or DICOM images and click run segmentation.
          No installation required and all data remains on your local workstation.
        </p>
      </div>
      <div class="mm-card-media">
        <video autoplay loop muted playsinline>
          <source src="{{ '/assets/images/video_neurodesk.mp4' | relative_url }}" type="video/mp4">
        </video>
      </div>
    </div>

  </div>
</div>