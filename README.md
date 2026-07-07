# Super-Resolution

### Environment Requirements

```bash
module load python/3.8.10
module load gcc gdal opencv cuda
pip install -r requirements_cc.txt
```

Increase spatial resolution of multispectral images using AI (ML/DL)

**10m Bands SR model**

* [ESRGAN-PA](./ESRGAN-PA/)


**20m Bands SR model**

* [RefGLEAN](./RefGLEAN/)
* [LightRefGLEAN](./LightRefGLEAN/)

**All Bands SR model**

* [SenGLEAN](./SenGLEAN/)
* [LightSenGLEAN](./LightSenGLEAN/)

For evaluation the model on a new dataset first perform target adaptation using each of the models as instructed in their respective ***./Testing*** directory and save the checkpoints.

Then you can perform the inference using the [Inference Notebook](./Inference.ipynb)

> **_NOTE_:** During inference the bands of sentinel input must be [B2, B3, B4, B8] @ 10m  + [ B5, B6, B7, B8a ] (interpolated to 10m)
