from __future__ import annotations
import json, sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat, UnidentifiedImageError
from .acne_model import model_status, predict_acne

DB_PATH=Path(__file__).resolve().parent.parent/"skinora.db"
MAX_BYTES=8*1024*1024
app=FastAPI(title="Skinora API",description="Facial wellness MVP: image quality checks and questionnaire-guided recommendations.",version="2.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:4173","http://127.0.0.1:4173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
SKIN_TYPES={"Oily","Dry","Combination","Normal","Sensitive","Not sure"}
CONCERNS={"Acne & blemishes","Dark spots / pigmentation","Uneven texture","Facial puffiness","Dryness","Oiliness","Redness / sensitivity"}

def connect():
 c=sqlite3.connect(DB_PATH);c.row_factory=sqlite3.Row;return c
def init_db():
 with connect() as db:
  db.execute("CREATE TABLE IF NOT EXISTS assessments (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, skin_type TEXT NOT NULL, age_range TEXT NOT NULL, concerns_json TEXT NOT NULL, sensitivity TEXT NOT NULL, concern_score INTEGER NOT NULL, photo_quality_json TEXT NOT NULL, profile_json TEXT NOT NULL, routine_json TEXT NOT NULL, recommendations_json TEXT NOT NULL, disclaimer TEXT NOT NULL)")
  db.commit()
@app.on_event("startup")
def startup():init_db()
@app.get("/")
def root():return {"app":"Skinora API","status":"running","docs":"/docs"}
@app.get("/api/health")
def health():return {"status":"ok","database":"sqlite","photo_storage":"disabled","acne_model":model_status()}

@app.get("/api/model-status")
def get_model_status():return model_status()

def quality_check(data):
 try:
  im=Image.open(BytesIO(data));im.verify();im=Image.open(BytesIO(data)).convert("RGB")
 except (UnidentifiedImageError,OSError,ValueError):
  raise HTTPException(400,"The uploaded file is not a readable image.")
 w,h=im.size
 if min(w,h)<160:return {"uploaded":True,"width":w,"height":h,"brightness":None,"sharpness_proxy":None,"quality":"Needs a larger image","summary":"Image is small; try at least 160 × 160 pixels.","checks":{"resolution":"low","lighting":"not assessed","sharpness":"not assessed","face_alignment":"not assessed"}}
 im.thumbnail((256,256));g=im.convert("L");mean=round(ImageStat.Stat(g).mean[0],1);px=list(g.getdata());ww,hh=g.size;diff=[]
 for y in range(hh-1):
  for x in range(ww-1):
   i=y*ww+x
   if i%5==0:diff.append((abs(px[i]-px[i+1])+abs(px[i]-px[i+ww]))/2)
 sharp=round(sum(diff)/max(1,len(diff)),2);light="too dark" if mean<55 else "very bright" if mean>205 else "within broad range";focus="possibly blurry" if sharp<4 else "roughly adequate";issues=[]
 if light!="within broad range":issues.append("Lighting may be too dark or bright; try even indirect daylight.")
 if focus=="possibly blurry":issues.append("Image may be blurry; hold the camera steady and refocus.")
 if min(w,h)<500:issues.append("A larger image may improve quality checks.")
 summary=" ".join(issues) if issues else "Brightness and rough sharpness look broadly acceptable. Face alignment is not checked."
 return {"uploaded":True,"width":w,"height":h,"brightness":mean,"sharpness_proxy":sharp,"quality":"Review photo" if issues else "Basic quality looks acceptable","summary":summary,"checks":{"resolution":"small" if min(w,h)<500 else "adequate dimensions","lighting":light,"sharpness":focus,"face_alignment":"not assessed"}}

def plan(skin,concerns,sensitivity):
 sensitive=skin=="Sensitive" or sensitivity=="Often" or "Redness / sensitivity" in concerns;oily=skin in {"Oily","Combination"}
 profile={"title":"Balanced basics" if skin=="Not sure" else f"{skin} skin, gently supported","summary":"Start with a simple routine and observe how your skin feels." if skin=="Not sure" else f"Guidance based on reported {skin.lower()} skin and selected concerns; not a diagnosis.","tags":[skin,*concerns[:3]]}
 am=["Gentle cleanser, or rinse with water if cleansing leaves skin tight.","Lightweight non-comedogenic moisturizer." if oily else "Moisturizer suited to your skin's comfort.","Broad-spectrum sunscreen SPF 30+ every morning; reapply as directed."]
 if sensitive:am=["Fragrance-free gentle cleanser; avoid scrubs and hot water.","Fragrance-free moisturizer that supports the skin barrier.",am[2]]
 pm=["Gentle cleanser to remove sunscreen, makeup and daily buildup."]
 if "Acne & blemishes" in concerns and not sensitive:pm.append("Optional: ask a qualified professional about low-strength salicylic acid; introduce slowly and stop if irritated.")
 if "Dark spots / pigmentation" in concerns and not sensitive:pm.append("Sun protection is important; ask a qualified professional about options such as niacinamide or azelaic acid.")
 pm.append("Moisturizer; keep the routine consistent and comfortable.")
 if sensitive:pm.append("Patch-test new products and avoid anything that stings or worsens redness.")
 bank={"Acne & blemishes":("Acne support","Avoid picking. A clinician or pharmacist can discuss evidence-based options for persistent acne."),"Dark spots / pigmentation":("Sun protection","Broad-spectrum SPF 30+ helps prevent uneven pigmentation from becoming darker. New or changing spots should be assessed."),"Uneven texture":("Gentle consistency","Prioritize gentle cleansing, moisturizer and sunscreen before adding targeted products."),"Facial puffiness":("Observe possible triggers","Note sleep and allergy patterns. Persistent or sudden swelling needs medical assessment."),"Dryness":("Barrier-supporting moisturizer","Glycerin or ceramides may support skin hydration."),"Oiliness":("Lightweight hydration","Oily skin can still benefit from a comfortable non-comedogenic moisturizer."),"Redness / sensitivity":("Fragrance-free basics","Keep the routine short; persistent redness merits professional assessment.")}
 rec=[]
 for c in concerns:
  if c in bank:
   name,reason=bank[c]
   if sensitive and c in {"Acne & blemishes","Uneven texture"}:name,reason="Gentle barrier care","Prioritize comfort and professional guidance before exfoliating actives."
   rec.append({"concern":c,"name":name,"reason":reason})
 if not rec:rec=[{"concern":"General","name":"Gentle basics","reason":"Start with cleanser, moisturizer and sunscreen."}]
 disclaimer="Educational guidance, not diagnosis or treatment. If a trained model checkpoint is installed, ResNet50 provides a research-only acne-severity estimate fine-tuned on ACNE04; it is not a medical diagnosis and may not generalize to real-world selfies. Photo-quality checks do not detect pigmentation, disease or puffiness. Seek a dermatologist for persistent, painful, changing or worsening symptoms."
 return {"profile":profile,"routine":{"morning":am,"evening":pm},"recommendations":rec,"disclaimer":disclaimer}

@app.post("/api/analyze")
async def analyze(skin_type:str=Form(...),age_range:str=Form("18–25"),concerns:str=Form("[]"),sensitivity:str=Form("Sometimes"),severity:int=Form(2),photo:Optional[UploadFile]=File(None)):
 if skin_type not in SKIN_TYPES:raise HTTPException(422,"Please choose a valid skin type.")
 try:selected=json.loads(concerns)
 except (json.JSONDecodeError,TypeError):raise HTTPException(422,"Concerns must be a JSON list.")
 if not isinstance(selected,list):raise HTTPException(422,"Concerns must be a JSON list.")
 selected=list(dict.fromkeys(x for x in selected if x in CONCERNS))
 if sensitivity not in {"Rarely","Sometimes","Often"}:sensitivity="Sometimes"
 severity=max(1,min(5,int(severity)))
 pq={"uploaded":False,"quality":"No photo","summary":"No photo uploaded; questionnaire-only guidance.","checks":{"resolution":"not assessed","lighting":"not assessed","sharpness":"not assessed","face_alignment":"not assessed"}}
 acne_prediction={"available":False,"status":"no_photo","message":"Upload a photo to request an acne-severity prediction. A trained checkpoint must be installed first."}
 if photo:
  if photo.content_type not in {"image/jpeg","image/png","image/webp"}:raise HTTPException(415,"Upload a JPG, PNG or WEBP image.")
  data=await photo.read()
  if not data:raise HTTPException(400,"Uploaded image is empty.")
  if len(data)>MAX_BYTES:raise HTTPException(413,"Image must be 8 MB or smaller.")
  pq=quality_check(data)
  acne_prediction=predict_acne(data)
 p=plan(skin_type,selected,sensitivity);created=datetime.now(timezone.utc).isoformat(timespec="seconds")
 with connect() as db:
  cur=db.execute("INSERT INTO assessments(created_at,skin_type,age_range,concerns_json,sensitivity,concern_score,photo_quality_json,profile_json,routine_json,recommendations_json,disclaimer) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(created,skin_type,age_range,json.dumps(selected),sensitivity,severity,json.dumps(pq),json.dumps(p["profile"]),json.dumps(p["routine"]),json.dumps(p["recommendations"]),p["disclaimer"]))
  db.commit();aid=cur.lastrowid
 return {"id":aid,"created_at":created,"skin_type":skin_type,"age_range":age_range,"concerns":selected,"sensitivity":sensitivity,"concern_score":severity,"photo_quality":pq,"acne_prediction":acne_prediction,**p}
@app.get("/api/assessments")
def assessments():
 with connect() as db:rows=db.execute("SELECT id,created_at,skin_type,age_range,concerns_json,sensitivity,concern_score,photo_quality_json FROM assessments ORDER BY id DESC LIMIT 100").fetchall()
 out=[]
 for row in rows:
  x=dict(row);x["concerns"]=json.loads(x.pop("concerns_json"));x["photo_quality"]=json.loads(x.pop("photo_quality_json"));out.append(x)
 return out
@app.get("/api/assessments/{aid}")
def assessment(aid:int):
 with connect() as db:row=db.execute("SELECT * FROM assessments WHERE id=?",(aid,)).fetchone()
 if row is None:raise HTTPException(404,"Assessment not found.")
 x=dict(row)
 for k in ["concerns_json","photo_quality_json","profile_json","routine_json","recommendations_json"]:x[k.replace("_json","")]=json.loads(x.pop(k))
 return x
