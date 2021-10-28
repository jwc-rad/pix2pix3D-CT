# pix2pix3D-CT
<p>Keras implementation of modified pix2pix with 3D convolutions. Developed for CT data.</p>
<p>This repository contains the source code of the following paper:</p>
<blockquote>
  <p>
    <strong>Generating Synthetic Contrast Enhancement from Non-contrast Chest Computed Tomography Using a Generative Adversarial Network</strong>
    <br>
    <i>Scientific Reports</i> 2021 Oct 14;11(1):20403. <a href="https://doi.org/10.1038/s41598-021-00058-3">doi: 10.1038/s41598-021-00058-3.</a>
  </p>
</blockquote>

## Notes
<p>To try out your own training and inference, each case in the data set should contain a pair of stack of axial CT scans in DICOM format. Please refer to <code>train.ipynb</code> and <code>inference.ipynb</code> for details.</p>
<p>Although this project was developed for CT data, the pix2pix3D network can work for any type of input data, if <code>source/data_loader.py</code> is properly modified.</p>
