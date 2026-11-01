SKINORA — RESNET50 + ACNE04 UPDATE
==================================

WHAT CHANGED
------------
- Added a ResNet50 inference module for four-level acne-severity estimates.
- Added a training script that downloads ACNE04 from Hugging Face and fine-tunes
  an ImageNet-1K-pretrained ResNet50.
- Added model-status API endpoint and frontend status/prediction display.
- Keeps questionnaire-based skincare guidance, SQLite history, basic photo
  quality checks, dermatologist search and care-approach information.
- If the model has not been trained, the app explicitly says no AI prediction
  was made. It never fabricates a model result.

IMPORTANT DATASET / MODEL FACTS
-------------------------------
1. ACNE04 is the primary task dataset: 1,457 facial acne photographs with four
   severity levels (mild, moderate, severe, very severe). The Hugging Face card
   preserves the original authors' annotations and states academic/research use
   only. Do not use it commercially without permission from the original authors.
2. ImageNet-1K is the additional dataset used indirectly through torchvision's
   pretrained ResNet50 weights. The network starts with generic visual features
   learned from ImageNet-1K and is fine-tuned on ACNE04. The images are not
   bundled in this project; the pretrained weights are downloaded on training.
3. Other dermatology datasets such as HAM10000/ISIC contain dermoscopic skin-lesion
   images and diagnosis labels, not the same four acne-severity labels. They are
   NOT mixed into ACNE04 training because doing so would create incorrect labels
   and domain mismatch. They could be used later for a separate, explicitly
   separated lesion-research task, not for this selfie acne score.
4. The optional --extra-data-dir argument supports additional acne datasets only
   if their labels genuinely map to the same four classes and their license allows
   use. Put them into this structure:
       extra-data\mild\image1.jpg
       extra-data\moderate\image2.jpg
       extra-data\severe\image3.jpg
       extra-data\very_severe\image4.jpg
   Do not relabel datasets based on guesses. Keep dataset provenance and licenses.
5. A trained model is not included in this zip. Training must be run locally to
   create backend\models\acne_resnet50.pth. Until then the frontend reports that
   the model needs training. No prediction is made.
6. This is a student research prototype, not a medical device. ACNE04 is small and
   may not generalize to everyday selfies, different lighting, or all skin tones.
   Do not claim clinical accuracy from one train run. Report held-out test metrics.

PROJECT PATHS (Windows)
-----------------------
Assumed project folder: C:\Users\Aakanksha\Downloads\Skinora\

BACKEND SETUP
1. Open a terminal in the backend folder:
   cd /d C:\Users\Aakanksha\Downloads\Skinora\backend
2. Install dependencies (uses the existing venv in the Skinora root):
   ..\venv\Scripts\python.exe -m pip install -r requirements.txt
3. Start FastAPI:
   ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload
4. API docs: http://127.0.0.1:8000/docs
5. Model status: http://127.0.0.1:8000/api/model-status

TRAIN THE MODEL (run once; internet required for dataset + ImageNet weights)
1. Open a SECOND terminal:
   cd /d C:\Users\Aakanksha\Downloads\Skinora\backend
2. Start with the default settings:
   ..\venv\Scripts\python.exe training\train_acne_model.py --epochs 8 --batch-size 16
3. First run downloads the ACNE04 dataset (about 1.1 GB) and pretrained ResNet50
   weights. GPU/CUDA is recommended; CPU training can take a long time.
4. To include extra compatible acne images:
   ..\venv\Scripts\python.exe training\train_acne_model.py --epochs 8 --batch-size 16 --extra-data-dir C:\path\to\extra-data
5. When training finishes, check:
   backend\models\acne_resnet50.pth
   backend\models\training_metrics.json
6. Restart FastAPI if needed, refresh the web app, and check /api/model-status.

If training fails due to package/CUDA compatibility, install the matching PyTorch
and torchvision build for your Python and hardware from https://pytorch.org/get-started/locally/.
The script uses num_workers=0 by default for Windows compatibility.

FRONTEND
1. Open another terminal:
   cd /d C:\Users\Aakanksha\Downloads\Skinora\frontend
2. Start Vite:
   npm run dev
3. Open the URL printed by Vite (usually http://localhost:5173).

FILES TO COPY
-------------
Copy frontend/src/App.jsx, App.css and index.css into your existing frontend/src/.
Copy backend/app/main.py, backend/app/acne_model.py, backend/app/__init__.py,
backend/training/train_acne_model.py and backend/requirements.txt into your backend.
The zip includes all of these files and this README.

DATA SOURCES / REFERENCES
-------------------------
- ACNE04 dataset (academic/research use only):
  https://huggingface.co/datasets/Layered-Labs/ACNE04
- Original ACNE04 paper: Wu et al., “Joint acne image grading and counting via
  label distribution learning,” ICCV 2019.
- ImageNet-1K pretrained ResNet50 weights (downloaded via torchvision):
  https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
- ISIC Archive datasets (separate skin-lesion task; per-dataset licenses vary):
  https://challenge.isic-archive.com/data/
- HAM10000 reference (dermoscopic pigmented-lesion images; not an acne severity
  dataset and not combined with ACNE04 here):
  https://doi.org/10.7910/DVN/DBW86T

CURRENT LIMITATIONS
-------------------
- The trained checkpoint and training metrics are not included; you must train it.
- ResNet50 only estimates ACNE04's four acne-severity categories; it does not
  predict skin type, pigmentation, puffiness, cancer, or other skin diseases.
- Confidence is a model softmax score, not a calibrated probability of correctness.
- No authentication/multi-user separation. Do not expose this student app publicly.
- Uploaded photos are processed in memory and are not stored; assessment metadata
  is saved locally in SQLite.
