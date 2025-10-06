import os
import jwt
import uuid
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline, BitsAndBytesConfig
from diffusers import DiffusionPipeline
from langchain.agents import initialize_agent, Tool
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import ee
import whisper
from gtts import gTTS
import requests
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image
import io
import numpy as np
from sklearn.linear_model import LogisticRegression
from nowpayments import NOWPayments
import base64
import torch

app = FastAPI()
app.mount("/videos", StaticFiles(directory="videos"), name="videos")  # Serve saved videos
security = HTTPBearer()

# Env Vars
MAILJET_API_KEY = os.getenv("MAILJET_API_KEY")
MAILJET_SECRET_KEY = os.getenv("MAILJET_SECRET_KEY")
NOWPAYMENTS_API_KEY = os.getenv("NOWPAYMENTS_API_KEY")
NOWPAYMENTS_IPN_SECRET = os.getenv("NOWPAYMENTS_IPN_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-prod")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./socio.db")
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://your-frontend.koyeb.app")
PLAN_ID = os.getenv("NOWPAYMENTS_PLAN_ID")

ALGORITHM = "HS256"
nowpayments = NOWPayments(NOWPAYMENTS_API_KEY, sandbox=False)  # Production: False

# DB Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models (added user_id FK)
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    trial_end = Column(DateTime, nullable=True)
    is_subscribed = Column(Boolean, default=False)

class UserSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    message = Column(Text)
    response = Column(Text)
    sentiment = Column(Float)
    lesson = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserVideo(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    path = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserResponse(BaseModel):
    id: int
    email: str
    is_subscribed: bool
    trial_active: bool
    class Config:
        orm_mode = True

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# JWT, Subscription Check (unchanged from prior)
def create_jwt(data: dict, expires_delta: timedelta = timedelta(minutes=30)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = verify_jwt(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def check_subscription(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    trial_active = user.trial_end and now < user.trial_end
    if not (trial_active or user.is_subscribed):
        raise HTTPException(status_code=402, detail="Subscription required. Trial expired.")
    return user

# Mailjet Magic Link
def send_magic_link(email: str, token: str):
    url = f"{FRONTEND_URL}/verify?token={token}"
    body = f"Click to login to Socio.AI: {url}"
    auth = (MAILJET_API_KEY, MAILJET_SECRET_KEY)
    data = {
        "Messages": [{
            "From": {"Email": FROM_EMAIL, "Name": "Socio.AI"},
            "To": [{"Email": email}],
            "Subject": "Your Magic Login Link",
            "TextPart": body
        }]
    }
    requests.post("https://api.mailjet.com/v3.1/send", auth=auth, json=data)

@app.post("/send-magic-link")
async def send_magic_link_endpoint(email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email, trial_end=datetime.utcnow() + timedelta(days=3))
        db.add(user)
        db.commit()
        db.refresh(user)
    magic_jwt = create_jwt({"sub": email}, timedelta(minutes=15))
    send_magic_link(email, magic_jwt)
    return {"message": "Link sent to your email"}

@app.get("/verify")
async def verify_magic_link(token: str = Query(...), db: Session = Depends(get_db)):
    try:
        payload = verify_jwt(token)
        email = payload["sub"]
        user = db.query(User).filter(User.email == email).first()
        if user:
            access_token = create_jwt({"sub": email})
            return JSONResponse({"access_token": access_token, "user": UserResponse.from_orm(user)})
        raise HTTPException(status_code=401)
    except:
        raise HTTPException(status_code=401, detail="Invalid/expired link")

# Email Reminder
def send_email(to_email: str, subject: str, body: str):
    msg = MimeMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MimeText(body, 'plain'))
    server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(FROM_EMAIL, to_email, msg.as_string())
    server.quit()

# Subscription & IPN (production, redirect)
@app.post("/subscribe")
async def create_subscription(user: User = Depends(get_current_user)):
    if not PLAN_ID:
        raise HTTPException(status_code=500, detail="Plan not configured")
    subscription = nowpayments.create_subscription(
        subscription_plan_id=PLAN_ID,
        customer_email=user.email,
        success_url=f"{FRONTEND_URL}/dashboard"
    )
    return {"subscription_id": subscription['subscription_id'], "status": subscription['status']}

@app.post("/ipn")
async def handle_ipn(request: dict):
    if request.get('ipn_secret') != NOWPAYMENTS_IPN_SECRET:
        raise HTTPException(status_code=401)
    subscription_id = request.get('subscription_id')
    payment_status = request.get('payment_status')
    if subscription_id:
        db = SessionLocal()
        user_email = request.get('customer_email')
        user = db.query(User).filter(User.email == user_email).first()
        if user:
            if payment_status == 'confirmed':
                user.is_subscribed = True
                user.trial_end = None
            elif payment_status == 'expired':
                user.is_subscribed = False
                renew_link = f"{FRONTEND_URL}/subscribe"
                body = f"Your subscription expired. Renew: {renew_link}"
                send_email(user.email, "Renew Subscription", body)
            db.commit()
        db.close()
    return {"status": "ok"}

# Models & Functions (quantized, interactive agent - full from prior, abridged for space; ensure torch, etc.)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

model_name_llama = "meta-llama/Llama-3.2-11B-Vision"
processor_llama = AutoProcessor.from_pretrained(model_name_llama)
model_llama = AutoModelForVision2Seq.from_pretrained(model_name_llama, quantization_config=quantization_config, device_map="auto")

wan_model_id = "Wan-AI/Wan2.1-T2V-14B"
wan_pipe = DiffusionPipeline.from_pretrained(wan_model_id, torch_dtype=torch.float16, use_safetensors=True)
wan_pipe.enable_model_cpu_offload()

mmaudio_model_id = "hkchengrex/MMAudio"
mmaudio_processor = AutoProcessor.from_pretrained(mmaudio_model_id)
mmaudio_model = AutoModelForCausalLM.from_pretrained(mmaudio_model_id, quantization_config=quantization_config, device_map="auto")

whisper_model = whisper.load_model("medium").half()

pipe = pipeline("text-generation", model=model_llama, tokenizer=processor_llama.tokenizer, max_new_tokens=200, temperature=0.7, device_map="auto")
llm = HuggingFacePipeline(pipeline=pipe)

ee.Initialize()

# GEE, Video Gen, Sound, Transcribe, TTS (unchanged)
def fetch_gee_video(location, start_date='2024-01-01', end_date='2025-10-01'):
    point = ee.Geometry.Point(location[1], location[0])
    collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterBounds(point.buffer(10000)) \
        .filterDate(start_date, end_date) \
        .sort('CLOUD_COVER') \
        .map(lambda image: image.visualize(**{'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0, 'max': 30000}))
    params = {'region': point.buffer(5000).bounds().getInfo()['coordinates'], 'dimensions': 720, 'format': 'mp4', 'framesPerSecond': 5}
    url = collection.getVideoThumbURL(params)
    response = requests.get(url)
    video_path = "gee_4d_video.mp4"
    with open(video_path, 'wb') as f:
        f.write(response.content)
    return video_path

def generate_geo_video(user_text, location, state="confident", mode="casual", start_date='2024-01-01', end_date='2025-10-01'):
    gee_video = fetch_gee_video(location, start_date, end_date)
    clip = VideoFileClip(gee_video)
    ref_image = Image.fromarray(np.uint8(clip.get_frame(0) * 255))
    prompt = f"A realistic 4D {state} {mode} social interaction evolving over time in {location}: {user_text}."
    video_frames = wan_pipe(prompt, image=ref_image, num_inference_steps=30, guidance_scale=7.5).frames[0]
    clip = ImageSequenceClip(video_frames, fps=8)
    video_path = f"videos/{uuid.uuid4()}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    clip.write_videofile(video_path, codec="libx264")
    return video_path

def generate_sound_effects(video_path, user_text, state="confident", mode="casual"):
    prompt = f"Sound effects for {state} {mode} interaction: {user_text}."
    inputs = mmaudio_processor(text=prompt, videos=[video_path], return_tensors="pt", padding=True)
    inputs = {k: v.to(mmaudio_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        audio_outputs = mmaudio_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8)
    audio = mmaudio_processor.decode_audio(audio_outputs[0].cpu().numpy())
    from scipy.io.wavfile import write
    write("sound_effects.wav", 22050, (audio * 32767).astype(np.int16))
    video = VideoFileClip(video_path)
    audio_clip = AudioFileClip("sound_effects.wav").subclip(0, video.duration)
    final_video = video.set_audio(audio_clip)
    final_path = video_path.replace('.mp4', '_audio.mp4')
    final_video.write_videofile(final_path, codec="libx264")
    return final_path

def transcribe_audio(audio_data):
    result = whisper_model.transcribe(io.BytesIO(audio_data), fp16=False)
    return result["text"]

def generate_tts(text):
    tts = gTTS(text)
    audio_path = "feedback.mp3"
    tts.save(audio_path)
    return audio_path

def predict_next_lesson(user_id: int, db: Session):
    sessions = db.query(UserSession).filter(UserSession.user_id == user_id).all()
    if len(sessions) < 3:
        return "How's your day going? Where would you like to explore today?"
    X = np.array([[s.sentiment] for s in sessions[-10:]])
    y = np.array([1 if s.sentiment > 0 else 0 for s in sessions[-10:]])
    if len(X) > 1:
        model = LogisticRegression().fit(X, y)
        pred = model.predict([[0]])[0]
        return "Ok, let's go to the Atlantic iceberg for calm vibes." if pred == 0 else "Let's try a busy city for energy!"
    return "Where would you like to explore today?"

def agent_core(text, location=[40.7128, -74.0060], start_date='2024-01-01', end_date='2025-10-01', audio_data=None, user_id: int = None, db: Session = None, history=""):
    if audio_data:
        text = transcribe_audio(audio_data)
    sentiment_pipe = pipeline("sentiment-analysis")
    sentiment = sentiment_pipe(text)[0]
    state = "confident" if sentiment['label'] == 'POSITIVE' else "anxious"
    score = sentiment['score'] if state == "confident" else -sentiment['score']
    
    suggestion = predict_next_lesson(user_id, db) if db else "Let's chat!"
    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message(history + text)
    prompt_template = PromptTemplate(
        input_variables=["history", "text", "suggestion"],
        template="You are a friendly social coach. History: {history}. User: {text}. Be interactive and suggestive: {suggestion}. Output: Engaging response + custom lesson for progress."
    )
    full_prompt = prompt_template.format(history=history, text=text, suggestion=suggestion)
    feedback = llm(full_prompt)
    
    mode = "casual"
    video = generate_geo_video(text + " " + suggestion, location, state, mode, start_date, end_date)
    audio_video = generate_sound_effects(video, text, state, mode)
    tts_audio = generate_tts(feedback)
    
    if db and user_id:
        session = UserSession(user_id=user_id, message=text, response=feedback, sentiment=score, lesson=suggestion)
        db.add(session)
        video_entry = UserVideo(user_id=user_id, path=audio_video, description=text)
        db.add(video_entry)
        db.commit()
    
    return {"feedback": feedback, "video": audio_video, "tts": tts_audio, "suggestion": suggestion}

tools = [Tool(name="VoiceGeoAgent", func=lambda **kwargs: agent_core(**kwargs), description="Interactive agent")]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, memory=ConversationBufferMemory())

@app.post("/process")
async def process_scenario(
    text: str = Form(None),
    location: str = Form("40.7128,-74.0060"),
    start_date: str = Form("2024-01-01"),
    end_date: str = Form("2025-10-01"),
    audio: UploadFile = File(None),
    history: str = Form(""),
    user: User = Depends(check_subscription),
    db: Session = Depends(get_db)
):
    audio_data = await audio.read() if audio else None
    loc_list = [float(x) for x in location.split(',')]
    result = agent_core(text, loc_list, start_date, end_date, audio_data, user.id, db, history)
    return JSONResponse(result)

@app.get("/videos")
async def get_videos(user: User = Depends(check_subscription), db: Session = Depends(get_db)):
    videos = db.query(UserVideo).filter(UserVideo.user_id == user.id).all()
    return [{"path": v.path, "desc": v.description} for v in videos]

@app.get("/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str):
    path = f"videos/{file_id}" if file_type == "video" else "feedback.mp3"
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
