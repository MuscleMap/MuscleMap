---
title: Change log 
nav_order: 1
description: "Change log for MuscleMap code as released in main Github page" 
parent: Developer section
permalink: /change-log/
---

## Change log and model updates

Segmentation models in MuscleMap are periodically retrained on expanded and more diverse datasets to optimise performance, robustness, and cross-site generalisability.

All notable changes to **MuscleMap** are listed below.  
Source: [GitHub Releases](https://github.com/MuscleMap/MuscleMap/releases)

<!-- ===== VERSION 1.1 ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v1.2</strong>
  <span style="
    background: #2ea44f;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 8px;">
    Latest
  </span>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Included 10 more muscles (left and right pectineus, obturator externus, obturator internus, piriformis, 
      gemelli and quadratus femoris)<li>
  <li>Included more training data <li>
</ul>
</div>


<!-- ===== VERSION 1.0 ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v1.1</strong>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Trained whole-body model on a more diverse dataset to improve generalizability</li>
  <li>Now using foreground cropping to improve inference efficiency</li>
</ul>
</div>



<!-- ===== VERSION 1.0 ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v1.0</strong>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Added the whole-body segmentation model (v1.0)</li>
  <li>Improved handling of large images in <code>mm_segment</code></li>
  <li>Added new options to <code>mm_segment</code></li>
  <li>Cleaned and improved <code>mm_extract_metrics</code></li>
</ul>

</div>



<!-- ===== VERSION 0.3 PRE-RELEASE ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v0.3</strong>
  <span style="
    background: #ff9800;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 8px;">
    Pre-release
  </span>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Introduced version numbers for trained models</li>
  <li>Preparation for the v1.0 release</li>
</ul>

</div>



<!-- ===== VERSION 0.2 PRE-RELEASE ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v0.2</strong>
  <span style="
    background: #ff9800;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 8px;">
    Pre-release
  </span>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Added <code>mm_register_to_template</code> for anatomical registration</li>
</ul>

</div>



<!-- ===== VERSION 0.1 PRE-RELEASE ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v0.1</strong>
  <span style="
    background: #ff9800;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 8px;">
    Pre-release
  </span>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Initial public pre-release including:</li>
  <ul>
    <li><code>mm_segment</code></li>
    <li><code>mm_extract_metrics</code></li>
    <li><code>mm_gui</code></li>
  </ul>
</ul>

</div>



<!-- ===== VERSION 0.0 PRE-RELEASE ===== -->
<p align="left" style="margin-bottom: 4px;">
  <strong style="font-size: 22px;">v0.0</strong>
  <span style="
    background: #ff9800;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    margin-left: 8px;">
    Pre-release
  </span>
</p>

<div style="
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 24px;
  background: #ffffff;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);">

<ul>
  <li>Initial repository version with early segmentation tools</li>
</ul>

</div>

<a class="mm-btn mm-btn-ghost mm-btn-back"
   href="{{ '/' | relative_url }}">
  ← Back to MuscleMap overview
</a>
