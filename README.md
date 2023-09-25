# CheckID: Guilloche Detection for Identity Document authentication
Git repository for the CheckID project, carried out by Musab Al-Ghadi.

## Introduction <br />

Identity documents (IDs) are always including more and more sophisticated security features in their designs in order to ward off potential counterfeiters, fraudsters and impostors. One of these security features is the Guilloche. The Guilloche design is a pattern of computer-generated fine lines that forms a unique shape. The target is to develop detection and verification approach of the Guilloche pattern in order to ensure the authenticity of the identity documents.
<img
  src="blob/FrenchID.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  <br />
  
## Contents <br />

1- CFD_model <br />
This model employs an encoder-decoder-classifier sub-networks which enable the model to map the input image into a lower-dimension feature vector, and then to reconstruct the output image. The objective of classifier is to well classify the input image into a real or fake image. 
<img
  src="blob/CFD.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  <br />
2- FsAFD_model. <br /> 
This model is similar to the CFD_model, the ony difference is that the classifier network f(.) is replaced by a onstrained-adversarial model A(.).
<img
  src="blob/FsAFD.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  <br />

## Description of files in this repository <br />

- Codes

