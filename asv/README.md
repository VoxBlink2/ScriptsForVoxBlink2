## SV Evaluation
To avoid the complex process of downloading data, we provide some model checkpoints for evaluation and fine-tuning. We train the different models with the VoxBlink2 and then LMFT on the VoxCeleb2. Finally, the performances are evaluated on the Vox1-O test set. For more training details, please refer to our manuscript or contact us via e-mail.

<table>
  <tr>
    <td rowspan="2"></td>
    <td colspan="2" style='text-align:center'>Pretrain</td>
    <td colspan="2" style='text-align:center'>FineTune</td>
  </tr>
  <tr>
    <td>EER</td>
    <td>minDCF</td>
    <td>EER</td>
    <td>minDCF</td>
  </tr>
  <tr>
    <td>ECAPA-TDNN</td>
    <td>1.728</td>
    <td>0.177</td>
    <td>0.675</td>
    <td>0.053</td>
  </tr>
  <tr>
    <td>ResNet34-GSP</td>
    <td>1.728</td>
    <td>0.176</td>
    <td>0.553</td>
    <td>0.056</td>
  </tr>
  <tr>
    <td>ResNet100-GSP</td>
    <td>1.132</td>
    <td>0.113</td>
    <td>0.457</td>
    <td>0.049</td>
  </tr>
  <tr>
    <td>ResNet293-SimAM-ASP</td>
    <td>0.707</td>
    <td>0.075</td>
    <td>0.228</td>
    <td>0.013</td>
  </tr>
</table>
