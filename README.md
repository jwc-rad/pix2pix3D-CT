# pix2pix3D-CT
<p>Keras implementation of modified pix2pix with 3D convolutions. Developed for CT data.</p>
<p>This repository contains the source code of the following paper:</p>
<blockquote>
  <p>
    <strong>Generating Synthetic Contrast Enhancement from Non-contrast Chest Computed Tomography Using a Generative Adversarial Network</strong>
    <br>
    Submitted manuscipt
  </p>
</blockquote>

# Notes
<p>To try out your own training and inference, each case in the data set should contain a pair of stack of axial CT scans in DICOM format. Please refer to <code>train.ipynb</code> and <code>inference.ipynb</code> for details.</p>
<p>Although this project was developed for CT data, the pix2pix3D network can work for any type of input data, if <code>source/data_loader.py</code> is properly modified.</p>
